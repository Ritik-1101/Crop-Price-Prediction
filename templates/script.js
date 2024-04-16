document.getElementById('prediction-form').addEventListener('submit', function(e) {
    e.preventDefault(); 
  
    const productname = document.getElementById('productname').value;
    const year = document.getElementById('year').value;
    const month = document.getElementById('month').value;
  
    fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ product_name: productname, year, month }) 
    })
    .then(response => response.json())
    .then(data => {
        const resultDiv = document.getElementById('result');
        resultDiv.classList.remove('hidden'); 
        document.getElementById('predicted-price').textContent = data.predicted_price;
    });
  });
  