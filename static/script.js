document.addEventListener("DOMContentLoaded", function () {
    document.getElementById("predictForm").addEventListener("submit", function (event) {
        event.preventDefault(); // Prevent default form submission

        let formData = new FormData(this);

        fetch("/predict", {
            method: "POST",
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            document.getElementById("predictionResult").innerText = "Prediction: " + data.prediction;
        })
        .catch(error => console.error("Error:", error));
    });
});
