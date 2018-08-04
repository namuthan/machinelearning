let x_vals = [];
let y_vals = [];
let m, b;
const learningRate = 0.5;
const opt = tf.train.sgd(learningRate);

function setup() {
  createCanvas(600, 600);

  m = tf.variable(tf.scalar(random(1)));
  b = tf.variable(tf.scalar(random(1)));
}

function mousePressed() {
  let x = map(mouseX, 0, width, 0, 1);
  let y = map(mouseY, 0, height, 1, 0);
  x_vals.push(x);
  y_vals.push(y);
}

function draw() {
  tf.tidy(() => {
    const ys = tf.tensor1d(y_vals);
    opt.minimize(() => loss(predict(x_vals), ys));
  });

  // draw the points
  background(0);
  stroke(255);
  strokeWeight(8);
  for (let i = 0; i < x_vals.length; i++) {
    let px = map(x_vals[i], 0, 1, 0, width);
    let py = map(y_vals[i], 0, 1, height, 0);
    point(px, py);
  }

  const lineX = [0, 1];
  const predictions = tf.tidy(() => predict(lineX));
  let lineY = predictions.dataSync();
  predictions.dispose();

  let x1 = map(lineX[0], 0, 1, 0, width);
  let x2 = map(lineX[1], 0, 1, 0, width);

  let y1 = map(lineY[0], 0, 1, height, 0);
  let y2 = map(lineY[1], 0, 1, height, 0);

  strokeWeight(2);
  line(x1, y1, x2, y2);
}

function predict(input) {
  const tf_inputs = tf.tensor1d(input);
  const outputs = tf_inputs.mul(m).add(b);
  return outputs;
}

function loss(pred, labels) {
  return pred
    .sub(labels)
    .square()
    .mean();
}
