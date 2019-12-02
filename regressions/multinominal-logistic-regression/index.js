require("@tensorflow/tfjs-node");
const tf = require("@tensorflow/tfjs");
const LogisticRegression = require("./logistic-regression");
const plot = require("node-remote-plot");
const _ = require("lodash");
const mnist = require("mnist-data");

const mnistData = mnist.training(0, 10);
const features = mnist.data.images.values(image => _.flatMap(image));

const encodedLabels = mnistData.labels.values.map(label => {
  const row = new Array(10).fill(0);
});
