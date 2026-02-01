
const fileInput = document.getElementById('file-upload');
const canvas = document.getElementById('input-canvas');
const ctx = canvas.getContext('2d');
const outputText = document.getElementById('output-text');
const statusDiv = document.getElementById('status');
const uploadLabel = document.querySelector('label[for="file-upload"]');

// Initialize
window.addEventListener('DOMContentLoaded', async () => {
    uploadLabel.classList.add('disabled');
    uploadLabel.textContent = "Loading Models...";
    
    const success = await loadModels();
    
    if (success) {
        uploadLabel.classList.remove('disabled');
        uploadLabel.textContent = "Choose Image";
    } else {
        uploadLabel.textContent = "Error Loading Models";
    }
});

fileInput.addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    // Display image
    const img = new Image();
    const url = URL.createObjectURL(file);
    
    img.onload = async () => {
        // Fit canvas to image but max width/height
        // Display logic only
        const maxWidth = 800;
        const maxHeight = 300;
        let w = img.width;
        let h = img.height;
        
        // Scale for display
        const ratio = Math.min(maxWidth / w, maxHeight / h);
        const displayW = w * ratio;
        const displayH = h * ratio;
        
        canvas.width = displayW;
        canvas.height = displayH;
        ctx.drawImage(img, 0, 0, displayW, displayH);
        
        outputText.value = "Processing...";
        
        try {
            // Predict (pass original img element to preserve resolution)
            const latex = await predictFormula(img);
            outputText.value = latex;
            updateStatus("Success!");
        } catch (error) {
            console.error(error);
            outputText.value = "Error: " + error.message;
        } finally {
            URL.revokeObjectURL(url);
        }
    };
    
    img.src = url;
});
