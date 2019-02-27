/**
 *                                                                   _
 *       _____  _                           ____  _                 |_|
 *      |  _  |/ \   ____  ____ __ ___     / ___\/ \   __   _  ____  _
 *      | |_| || |  / __ \/ __ \\ '_  \ _ / /    | |___\ \ | |/ __ \| |
 *      |  _  || |__. ___/. ___/| | | ||_|\ \___ |  _  | |_| |. ___/| |
 *      |_/ \_|\___/\____|\____||_| |_|    \____/|_| |_|_____|\____||_|
 *
 *      ================================================================
 *                 More than a coder, More than a designer
 *      ================================================================
 *
 *      - Document: index.js
 *      - Author: aleen42
 *      - Description: the main entry of this training project
 *      - Create Time: Feb 18th, 2019
 *      - Update Time: Feb 18th, 2019
 *
 */

const data = require('./data');
const tf = require('@tensorflow/tfjs');

/** disable deprecated warning */
tf.disableDeprecationWarnings();
/** parameters to be calculated during training with random value initialized */
const e = tf.variable(tf.scalar(1.0));

let _monitor;
let _count = 0;
(_monitor = () => console.log(`${_count++}: e=${e.dataSync()[0]}`))();

const _predict = x => tf.tidy(() => e.pow(x));
/** using MSE (Mean Squared Error) as an loss predicting methods */
const _loss = (predictions, labels) => predictions.sub(labels).square().mean();

/** using SGD (Stochastic Gradient Descent) as the optimizer during training */
/** higher learning rate results in higher lost */
/** lower learning rate results in slow learning */
const _optimizer = tf.train.sgd(0.0001/** learning rate */);

const _train = async (x, y, iterations = 75) => {
    for (let i = 0; i < iterations; i++) {
        _optimizer.minimize(() => _loss(_predict(x), y));
        /** print out predictions */
        _monitor();
        /** use tf.nextFrame to avoid blocking */
        await tf.nextFrame();
    }
};

(async () => _train(tf.tensor1d(data.map(([x]) => x)), tf.tensor1d(data.map(([, y]) => y)), data.length))();
