import React, { useEffect, useRef, useState } from "react";

import { Button, Heading, Link, useToast } from "@chakra-ui/react";
import * as tf from "@tensorflow/tfjs";
import { Bar } from "react-chartjs-2";
import { ExternalLinkIcon } from '@chakra-ui/icons'

import drawHere from "./drawHere.jpg";
import "./App.css";

const App = () => {
  const canvasState = useRef(false);
  const canvasRef = useRef(null);
  const denseModel = useRef();
  const CNNModel = useRef();

  const [densePreds, setDensePreds] = useState([]);
  const [cnnPreds, setCnnPreds] = useState([]);

  const toast = useToast();

  const handleStartDraw = () => {
    canvasState.current = true;
  };

  const handleEndDraw = (context) => {
    canvasState.current = false;
    context.beginPath();
    makePredictionDense();
    makePredictionCNN();
  };

  const handleDraw = (event, context) => {
    if (!canvasState.current) return;
    context.lineWidth = 5;
    context.lineCap = "round";
    context.strokeStyle = "#319795";
    context.lineTo(event.offsetX, event.offsetY);
    context.stroke();
    context.beginPath();
    context.moveTo(event.offsetX, event.offsetY);
  }

  const handleClear = () => {
    const context = canvasRef.current.getContext("2d");
    context.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
    setDensePreds([]);
    setCnnPreds([]);
  }

  useEffect(() => {
    tf.loadLayersModel(process.env.REACT_APP_DENSE_MODEL).then((loadedModel) => {
      denseModel.current = loadedModel;
      informAboutLoad("Dense")
    });

    tf.loadLayersModel(process.env.REACT_APP_CNN_MODEL).then((loadedModel) => {
      CNNModel.current = loadedModel;
      informAboutLoad("CNN")
    });

    const context = canvasRef.current.getContext("2d");
    canvasRef.current.height = 200;
    canvasRef.current.width = 200;
    canvasRef.current.addEventListener("mousedown", handleStartDraw);
    canvasRef.current.addEventListener("mouseup", () => handleEndDraw(context));
    canvasRef.current.addEventListener("mousemove", (event) => handleDraw(event, context))
    // eslint-disable-next-line
  }, []);

  const getImageTensorFromCanvas = () => {
    const context = canvasRef.current.getContext("2d");
    const imageData = context.getImageData(0, 0, 200, 200);
    const imageTensor = tf.browser.fromPixels(imageData, 1).cast("float32");
    const resizedTensor = tf.image.resizeBilinear(imageTensor, [28, 28]);
    return resizedTensor;
  }

  const makePredictionDense = () => {
    const imageTensor = getImageTensorFromCanvas();
    const reshapedTensor = imageTensor.reshape([1, 28, 28]);
    const predictionTensor = denseModel.current.predict(reshapedTensor);
    predictionTensor.flatten().data().then(setDensePreds).catch(informAboutModelLoadFail);
  }

  const makePredictionCNN = () => {
    const imageTensor = getImageTensorFromCanvas();
    const reshapedTensor = imageTensor.reshape([1, 28, 28, 1]);
    const predictionTensor = CNNModel.current.predict(reshapedTensor);
    predictionTensor.flatten().data().then(setCnnPreds).catch(informAboutModelLoadFail);
  }

  const informAboutLoad = (modelName) => {
    toast({
      title: `${modelName} model loaded succesfully!`,
      status: "success",
      duration: 5000,
      isClosable: true
    });
  }

  const informAboutModelLoadFail = (modelName) => {
    toast({
      title: `${modelName} model failed to load`,
      status: "error",
      duration: null,
      isClosable: false
    });
  }

  const data = {
    labels: ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
    datasets: [
      {
        label: 'CNN',
        data: cnnPreds,
        backgroundColor: '#6B46C1',
      },
      {
        label: 'Dense',
        data: densePreds,
        backgroundColor: '#4FD1C5',
      },
    ],
  }

  return (
    <div className="page">
      <Heading className="heading">Dense/CNN MNIST Comparision</Heading>
      <div className="wrapper">
        <div className="img_helper">
          <div className="canvas">
            <canvas ref={canvasRef} />
            <Button colorScheme="teal" onClick={handleClear}>Clear canvas</Button>
          </div>
          <img src={drawHere} width={100} alt="draw here" />
        </div>
        <div className="chart">
          <Bar data={data} height={100} width={300} />
        </div>
      </div>
      <div className="footer">
        <Link fontSize="xs" href="https://github.com/kamilbakierzynski" isExternal>
          Github <ExternalLinkIcon mx="2px" />
        </Link>
      </div>
    </div>
  )
}
export default App;