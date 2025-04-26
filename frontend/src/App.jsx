import { useState } from 'react';
import { useDropzone } from 'react-dropzone';
import './App.css';

function App() {
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [prediction, setPrediction] = useState(null);
  const [isLoading, setIsLoading] = useState(false);

  const onDrop = (acceptedFiles) => {
    setSelectedFiles(acceptedFiles);
    setPrediction(null); // Reset prediction when new file is uploaded
  };

  const handlePredict = async () => {
    if (!selectedFiles.length) {
      alert("Please upload an image to proceed.");
      return;
    }

    const formData = new FormData();
    selectedFiles.forEach((file) => {
      formData.append("file", file);
    });

    setIsLoading(true);

    try {
      const response = await fetch("http://localhost:5000/predict", {
        method: "POST",
        body: formData,
      });
      const data = await response.json();
      
      setTimeout(() => {
        setPrediction(data);
        setIsLoading(false); 
      }, 2000);
    } catch (error) {
      console.error("Error:", error);
      alert("Failed to make a prediction.");
      setIsLoading(false);
    }
  };

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: 'image/*',
    multiple: false,
  });

  return (
    <div className="app">
      <h1>Brain Tumor Detector 🧠</h1>

      <div
        className={`dropzone ${isDragActive ? 'active' : ''}`}
        {...getRootProps()}
      >
        <input {...getInputProps()} />
        {isDragActive ? (
          <p>Drop your image here...</p>
        ) : (
          <p>Drag & Drop your image or <span>Click to Upload</span></p>
        )}
      </div>

      <div className="image-preview">
        {selectedFiles.map((file, index) => (
          <img key={index} src={URL.createObjectURL(file)} alt="Preview" />
        ))}
      </div>

      <button className="predict-button" onClick={handlePredict} disabled={isLoading}>
        {isLoading ? "Processing..." : "Predict"}
      </button>

      {isLoading && (
        <div className="loading-icon">
          <div className="dots">
            <div></div>
            <div></div>
            <div></div>
          </div>
          Processing...
        </div>
      )}

      {prediction && !isLoading && (
        <div className="prediction-result">
          <h2>Predicted Class: {prediction.predicted_class}</h2>
          <h3>Confidence Scores:</h3>
          <ul>
            {Object.entries(prediction.confidence_scores).map(([label, score]) => (
              <li key={label}>
                {label}: {(score * 100).toFixed(2)}%
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

export default App;
