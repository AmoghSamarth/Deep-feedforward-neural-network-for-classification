export interface FnnLayer {
  name: string;
  weights: number[][]; // [inputDim, outputDim]
  biases: number[];    // [outputDim]
  activation: 'relu' | 'softmax';
  inputDim: number;
  outputDim: number;
  paramCount: number;
}

export interface ScalerConfig {
  mean: number[];
  scale: number[];
  var: number[];
  featureNames: string[];
  featureKeys: string[];
}

export interface FnnModelMetadata {
  modelName: string;
  course: string;
  student: string;
  usn: string;
  dateTrained: string;
  totalParameters: number;
  layers: Array<{
    name: string;
    units: number;
    activation: string;
    params?: number;
  }>;
}

export interface FnnModelArtifact {
  metadata: FnnModelMetadata;
  scaler: ScalerConfig;
  classNames: string[];
  layers: FnnLayer[];
}

export interface PredictionResult {
  predictedClass: string;
  predictedIndex: number;
  confidence: number; // 0 to 1
  probabilities: Record<string, number>;
  rawInput: number[];
  scaledInput: number[];
}

/**
 * Executes forward propagation through the Deep Feedforward Neural Network.
 * Mathematical equations:
 * Layer 1: z^[1] = W^[1] x_norm + b^[1], a^[1] = ReLU(z^[1])
 * Layer 2: z^[2] = W^[2] a^[1] + b^[2], a^[2] = ReLU(z^[2])
 * Layer 3: z^[3] = W^[3] a^[2] + b^[3], a^[3] = ReLU(z^[3])
 * Layer 4: z^[4] = W^[4] a^[3] + b^[4], y_hat = Softmax(z^[4])
 */
export function predictIrisSample(
  features: [number, number, number, number],
  model: FnnModelArtifact
): PredictionResult {
  const { scaler, layers, classNames } = model;

  // 1. StandardScaler Transformation: z = (x - mu) / sigma
  const scaledInput = features.map((val, idx) => (val - scaler.mean[idx]) / scaler.scale[idx]);

  // 2. Layer 1: Dense 16 + ReLU (Input 4 -> Output 16)
  const layer1 = layers[0];
  const a1 = new Array(layer1.outputDim).fill(0);
  for (let j = 0; j < layer1.outputDim; j++) {
    let sum = layer1.biases[j];
    for (let i = 0; i < layer1.inputDim; i++) {
      sum += scaledInput[i] * layer1.weights[i][j];
    }
    a1[j] = Math.max(0, sum); // ReLU
  }

  // 3. Layer 2: Dense 16 + ReLU (Input 16 -> Output 16)
  const layer2 = layers[1];
  const a2 = new Array(layer2.outputDim).fill(0);
  for (let j = 0; j < layer2.outputDim; j++) {
    let sum = layer2.biases[j];
    for (let i = 0; i < layer2.inputDim; i++) {
      sum += a1[i] * layer2.weights[i][j];
    }
    a2[j] = Math.max(0, sum); // ReLU
  }

  // 4. Layer 3: Dense 8 + ReLU (Input 16 -> Output 8)
  const layer3 = layers[2];
  const a3 = new Array(layer3.outputDim).fill(0);
  for (let j = 0; j < layer3.outputDim; j++) {
    let sum = layer3.biases[j];
    for (let i = 0; i < layer3.inputDim; i++) {
      sum += a2[i] * layer3.weights[i][j];
    }
    a3[j] = Math.max(0, sum); // ReLU
  }

  // 5. Layer 4: Dense 3 + Softmax (Input 8 -> Output 3)
  const layer4 = layers[3];
  const z4 = new Array(layer4.outputDim).fill(0);
  for (let j = 0; j < layer4.outputDim; j++) {
    let sum = layer4.biases[j];
    for (let i = 0; i < layer4.inputDim; i++) {
      sum += a3[i] * layer4.weights[i][j];
    }
    z4[j] = sum;
  }

  // Numerically stable Softmax: exp(z_i - max(z)) / sum(exp(z_j - max(z)))
  const maxZ = Math.max(...z4);
  const expZ = z4.map(z => Math.exp(z - maxZ));
  const sumExp = expZ.reduce((acc, val) => acc + val, 0);
  const probabilitiesArray = expZ.map(val => val / sumExp);

  // Determine predicted class
  let maxIdx = 0;
  let maxProb = probabilitiesArray[0];
  for (let i = 1; i < probabilitiesArray.length; i++) {
    if (probabilitiesArray[i] > maxProb) {
      maxProb = probabilitiesArray[i];
      maxIdx = i;
    }
  }

  const probMap: Record<string, number> = {};
  classNames.forEach((name, idx) => {
    probMap[name] = probabilitiesArray[idx];
  });

  return {
    predictedClass: classNames[maxIdx],
    predictedIndex: maxIdx,
    confidence: maxProb,
    probabilities: probMap,
    rawInput: features,
    scaledInput: scaledInput.map(v => Math.round(v * 10000) / 10000),
  };
}
