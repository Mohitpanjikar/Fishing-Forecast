import { ModelDefinition, ModelType } from "../types";
import { toast } from "sonner";

// API URL for ML backend
const API_BASE_URL = "http://localhost:5001/api";
let apiAvailable = false;

// Check if API is available
const checkApiAvailability = async () => {
  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 3000);
    
    const response = await fetch(`${API_BASE_URL}/health`, {
      signal: controller.signal
    });
    
    clearTimeout(timeoutId);
    
    if (response.ok) {
      console.log("ML API is available");
      apiAvailable = true;
    } else {
      console.warn("ML API returned error status");
      apiAvailable = false;
    }
  } catch (error) {
    console.warn("ML API is not available, will use fallback simulation", error);
    apiAvailable = false;
  }
};

// Run the check immediately
checkApiAvailability().catch(() => {
  console.warn("API availability check failed");
});

export const modelDefinitions: Record<ModelType, ModelDefinition> = {
  "Random Forest": {
    name: "Random Forest",
    description: "An ensemble of decision trees that improves accuracy and reduces overfitting through voting mechanisms.",
    accuracy: "80-90%",
    icon: "trees",
    color: "bg-green-600"
  },
  "SVM": {
    name: "Support Vector Machine",
    description: "A classifier that finds the optimal boundary to separate classes in high-dimensional space.",
    accuracy: "75-85%",
    icon: "divide",
    color: "bg-blue-600"
  },
  "Logistic Regression": {
    name: "Logistic Regression",
    description: "A simple yet effective statistical model for binary classification problems with probability outcomes.",
    accuracy: "70-80%",
    icon: "line-chart",
    color: "bg-indigo-600"
  },
  "Decision Tree": {
    name: "Decision Tree",
    description: "A tree-structured classifier that splits data based on feature importance and information gain.",
    accuracy: "65-75%",
    icon: "git-branch",
    color: "bg-yellow-600"
  },
  "KNN": {
    name: "K-Nearest Neighbors",
    description: "A distance-based classifier that makes predictions based on the closest examples in the feature space.",
    accuracy: "60-75%",
    icon: "target",
    color: "bg-red-600"
  },
  "Neural Network": {
    name: "Neural Network",
    description: "A deep learning model that mimics human brain structure to recognize complex patterns in data.",
    accuracy: "78-92%",
    icon: "network",
    color: "bg-purple-600"
  }
};

// Function to train model using Python backend
export const trainModel = async (
  model: ModelType,
  data: { X: number[][], y: number[] },
  params: Record<string, unknown>
): Promise<{ accuracy: number; confusionMatrix: number[][] }> => {
  // If we already know API is unavailable, use fallback immediately
  if (!apiAvailable) {
    console.log("Using fallback training immediately (API known to be unavailable)");
    return fallbackTrainModel(model, data);
  }

  try {
    // Reconstruct the data in the format expected by the API
    const trainingData = data.X.map((features, index) => ({
      lat: features[0],
      lon: features[1],
      hour: features[2],
      illegal: data.y[index]
    }));

    console.log(`Sending training request to ${API_BASE_URL}/train with ${trainingData.length} points`);

    // Send request to the Python backend
    const response = await fetch(`${API_BASE_URL}/train`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        model: model,
        data: trainingData,
      }),
    });

    console.log("API response status:", response.status);

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.error || 'Failed to train model');
    }

    const result = await response.json();
    console.log("Training result:", result);
    
    return {
      accuracy: result.accuracy,
      confusionMatrix: result.confusionMatrix
    };
  } catch (error) {
    console.error('Error training model:', error);
    // Record that API is unavailable for future calls
    apiAvailable = false;
    toast.error("Could not connect to ML backend, using simulation instead");
    // Fallback to simulation if API fails
    return fallbackTrainModel(model, data);
  }
};

// Fallback function in case API is unavailable
const fallbackTrainModel = (
  model: ModelType,
  data: { X: number[][], y: number[] }
): { accuracy: number; confusionMatrix: number[][] } => {
  console.warn('Using fallback training model - API unavailable');
  
  let baseAccuracy = 0;
  
  switch (model) {
    case "Random Forest":
      baseAccuracy = 0.85;
      break;
    case "Neural Network":
      baseAccuracy = 0.84;
      break;
    case "SVM":
      baseAccuracy = 0.82;
      break;
    case "Logistic Regression":
      baseAccuracy = 0.76;
      break;
    case "Decision Tree":
      baseAccuracy = 0.72;
      break;
    case "KNN":
      baseAccuracy = 0.68;
      break;
    default:
      baseAccuracy = 0.75;
  }
  
  const accuracy = baseAccuracy + (Math.random() * 0.1 - 0.05);
  
  const numSamples = data.y.length;
  const truePositives = Math.floor(numSamples * accuracy * 0.4);
  const trueNegatives = Math.floor(numSamples * accuracy * 0.6);
  const falsePositives = Math.floor(numSamples * (1 - accuracy) * 0.5);
  const falseNegatives = numSamples - truePositives - trueNegatives - falsePositives;
  
  return {
    accuracy,
    confusionMatrix: [
      [trueNegatives, falsePositives],
      [falseNegatives, truePositives]
    ]
  };
};

// Function to predict illegal fishing using Python backend
export const predictIllegalFishing = async (
  latitude: number,
  longitude: number,
  hour: number,
  modelType: ModelType = "Random Forest"
): Promise<{ result: boolean; probability: number }> => {
  // If we already know API is unavailable, use fallback immediately
  if (!apiAvailable) {
    console.log("Using fallback prediction immediately (API known to be unavailable)");
    return fallbackPrediction(latitude, longitude, hour);
  }

  try {
    console.log(`Sending prediction request to ${API_BASE_URL}/predict for location: ${latitude}, ${longitude}`);

    // Send request with timeout
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 3000);

    // Send request to the Python backend
    const response = await fetch(`${API_BASE_URL}/predict`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        model: modelType,
        lat: latitude,
        lon: longitude,
        hour: hour
      }),
      signal: controller.signal
    });

    clearTimeout(timeoutId);
    console.log("API prediction response status:", response.status);

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.error || 'Failed to get prediction');
    }

    const prediction = await response.json();
    console.log("Prediction result:", prediction);
    
    return {
      result: prediction.result,
      probability: prediction.probability
    };
  } catch (error) {
    console.error('Error predicting illegal fishing:', error);
    // Record that API is unavailable for future calls
    apiAvailable = false;
    // Fallback to simulation if API fails
    return fallbackPrediction(latitude, longitude, hour);
  }
};

// Fallback prediction in case API is unavailable
const fallbackPrediction = (
  latitude: number,
  longitude: number,
  hour: number
): { result: boolean; probability: number } => {
  console.warn('Using fallback prediction - API unavailable');
  
  const latEffect = Math.abs(latitude) / 90;
  const hourEffect = hour >= 19 || hour <= 4 ? 0.3 : -0.1;
  const randomFactor = Math.random() * 0.2;
  
  let probability = 0.3 + latEffect + hourEffect + randomFactor;
  probability = Math.max(0, Math.min(1, probability));
  
  return {
    result: probability > 0.5,
    probability
  };
};
