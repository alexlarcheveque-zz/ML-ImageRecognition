require("@tensorflow/tfjs-node");
const tf = require("@tensorflow/tfjs");
const LogisticRegression = require("./logistic-regression");
const plot = require("node-remote-plot");
const _ = require("lodash");
const mnist = require("mnist-data");

const mnistData = mnist.training(0, 1000);
const features = mnist.data.images.values(image => _.flatMap(image));

const encodedLabels = mnistData.labels.values.map(label => {
  const row = new Array(10).fill(0);
  row[label] = 1;
  return row;
});

const regression = new LogisticRegression(features, encodedLabels, {
  learningRate: 1,
  iterations: 5,
  bathSize: 100
});

regression.train();

const testMnistData = mnist.testing(0, 100);
const testFeatures = testMnistData.images.values(image => _.flatMap(image));
const testEncodedLabels = testMnistData.labels.map(label => {
  const row = new Array(10).fill(0);
  row[label] = 1;
  return row;
});

regression.test();
