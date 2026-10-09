<!DOCTYPE html>
<html lang="hi">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>D7 Korean Stock Manager</title>
  <style>
    body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f4f6f9; margin: 0; padding: 20px; }
    .container { max-width: 600px; margin: 0 auto; background: #fff; padding: 20px; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); }
    h2 { margin-top: 0; color: #333; text-align: center; }
    .product-title { font-size: 20px; font-weight: bold; color: #1a73e8; border-bottom: 2px solid #1a73e8; padding-bottom: 5px; margin-bottom: 15px; }
    
    .voice-section { display: flex; align-items: center; justify-content: space-between; background: #e8f0fe; padding: 12px 15px; border-radius: 8px; margin-bottom: 20px; }
    .mic-btn { background: #ea4335; color: white; border: none; padding: 10px 18px; border-radius: 50px; cursor: pointer; font-size: 15px; display: flex; align-items: center; gap: 8px; font-weight: bold; }
    .mic-btn:hover { background: #d93025; }
    
    .color-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(240px, 1fr)); gap: 12px; margin-bottom: 20px; }
    .color-card { display: flex; justify-content: space-between; align-items: center; padding: 12px 15px; border-radius: 8px; border: 1px solid #e0e0e0; background: #fafafa; }
    .color-name { font-weight: 600; font-size: 16px; color: #444; }
    
    /* Green Box for Available Stock */
    .stock-box.in-stock { background-color: #2e7d32; color: #ffffff; padding: 6px 14px; border-radius: 6px; font-weight: bold; font-size: 14px; }
    /* Red Box for NILL Stock */
    .stock-box.out-of-stock { background-color: #d32f2f; color: #ffffff; padding: 6px 14px; border-radius: 6px; font-weight: bold; font-size: 14px; }
    
    .share-btn { width: 100%; background: #25d366; color: white; border: none; padding: 14px; border-radius: 8px; font-size: 16px; font-weight: bold; cursor: pointer; text-align: center; }
    .share-btn:hover { background: #1eb956; }
  </style>
</head>
<body>

<div class="container">
  <h2>Stock Management Dashboard</h2>
  <div class="product-title">Item: D7 Korean</div>

  <!-- Voice Input Section -->
  <div class="voice-section">
    <span id="voice-status">Speak command (e.g. "D7 White 5 pkt")</span>
    <button class="mic-btn" onclick="startVoiceRecognition()">
      🎤 Speak
    </button>
  </div>

  <!-- Stock Grid with 10 Colors -->
  <div class="color-grid" id="stockGrid"></div>

  <!-- Send Report to Boss -->
  <button class="share-btn" onclick="shareToBoss()">
    📲 Send Summary to Boss (WhatsApp)
  </button>
</div>

<script>
  // Initial Stock Data for 10 Colors of D7 Korean
  const colorsData = [
    { name: "White", stock: 10 },
    { name: "Red", stock: 0 },
    { name: "Black", stock: 15 },
    { name: "Blue", stock: 0 },
    { name: "Green", stock: 8 },
    { name: "Yellow", stock: 25 },
    { name: "Pink", stock: 0 },
    { name: "Grey", stock: 12 },
    { name: "Maroon", stock: 5 },
    { name: "Navy Blue", stock: 0 }
  ];

  function renderStock() {
    const grid = document.getElementById('stockGrid');
    grid.innerHTML = '';

    colorsData.forEach((item, index) => {
      const card = document.createElement('div');
      card.className = 'color-card';

      const isAvailable = item.stock > 0;
      const stockClass = isAvailable ? 'in-stock' : 'out-of-stock';
      const stockText = isAvailable ? `${item.stock} Pkt` : 'NILL';

      card.innerHTML = `
        <span class="color-name">${item.name}</span>
        <span class="stock-box ${stockClass}" id="box-${index}">${stockText}</span>
      `;
      grid.appendChild(card);
    });
  }

  // Speech Recognition (Voice Input)
  function startVoiceRecognition() {
    const status = document.getElementById('voice-status');
    if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
      alert("Voice input is not supported in this browser. Try Google Chrome.");
      return;
    }

    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    const recognition = new SpeechRecognition();
    recognition.lang = 'hi-IN'; // Supports Hindi/English mix
    recognition.interimResults = false;

    status.innerText = "Listening... Speak now";

    recognition.onresult = function(event) {
      const transcript = event.results[0][0].transcript.toLowerCase();
      status.innerText = `Recognized: "${transcript}"`;
      processVoiceCommand(transcript);
    };

    recognition.onerror = function() {
      status.innerText = "Error recognizing voice. Try again.";
    };

    recognition.start();
  }

  // Parse Voice Command (e.g. "D7 white 5 pkt red 2 pkt")
  function processVoiceCommand(command) {
    colorsData.forEach(item => {
      const colorName = item.name.toLowerCase();
      if (command.includes(colorName)) {
        // Find numbers following the color name
        const regex = new RegExp(`${colorName}\\s*(\\d+)`);
        const match = command.match(regex);
        if (match && match[1]) {
          item.stock = parseInt(match[1]);
        }
      }
    });
    renderStock();
  }

  // Format and share report to Boss via WhatsApp
  function shareToBoss() {
    let report = "*Stock Report: D7 Korean*\n\n";
    colorsData.forEach(item => {
      if (item.stock > 0) {
        report += `✅ *${item.name}*: ${item.stock} Pkt\n`;
      } else {
        report += `❌ *${item.name}*: NILL\n`;
      }
    });

    const whatsappUrl = `https://wa.me/?text=${encodeURIComponent(report)}`;
    window.open(whatsappUrl, '_blank');
  }

  // Initial Render
  renderStock();
</script>

</body>
</html>
