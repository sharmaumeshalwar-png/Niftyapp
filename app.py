import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="D7 Korean Stock Manager", layout="centered")

html_code = """
<!DOCTYPE html>
<html lang="hi">
<head>
  <meta charset="UTF-8">
  <style>
    body { font-family: 'Segoe UI', sans-serif; background-color: #f4f6f9; margin: 0; padding: 10px; }
    .container { max-width: 600px; margin: 0 auto; background: #ffffff; padding: 20px; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); }
    .product-title { font-size: 18px; font-weight: bold; color: #1a73e8; border-bottom: 2px solid #1a73e8; padding-bottom: 8px; margin-bottom: 15px; text-align: center; }
    
    .catalog-img { width: 100%; border-radius: 8px; margin-bottom: 15px; border: 1px solid #ddd; }

    .color-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 10px; margin-bottom: 20px; }
    .color-card { display: flex; justify-content: space-between; align-items: center; padding: 10px 12px; border-radius: 8px; border: 1px solid #e0e0e0; background: #fafafa; }
    .color-name { font-weight: 600; font-size: 15px; color: #333333; }
    
    .stock-box.in-stock { background-color: #2e7d32; color: #ffffff; padding: 6px 12px; border-radius: 6px; font-weight: bold; font-size: 13px; }
    .stock-box.out-of-stock { background-color: #d32f2f; color: #ffffff; padding: 6px 12px; border-radius: 6px; font-weight: bold; font-size: 13px; }
    
    .share-btn { width: 100%; background: #25d366; color: white; border: none; padding: 14px; border-radius: 8px; font-size: 16px; font-weight: bold; cursor: pointer; text-align: center; }
  </style>
</head>
<body>

<div class="container">
  <div class="product-title">Item: D7 Premium Single Side Rose Gold Edge Frosted</div>
  
  <!-- Direct Image Link / Local Image Path -->
  <img src="d7_catalog.jpg" alt="D7 Shade Card" class="catalog-img" onerror="this.style.display='none'">

  <div class="color-grid" id="stockGrid"></div>

  <button class="share-btn" onclick="shareToBoss()">📲 Send Summary to Boss (WhatsApp)</button>
</div>

<script>
  const colorsData = [
    { name: "01 Dusty Pink", stock: 10 },
    { name: "02 Pink", stock: 0 },
    { name: "03 Khaki", stock: 15 },
    { name: "06 Wine", stock: 0 },
    { name: "07 Black", stock: 8 },
    { name: "08 Grey", stock: 25 },
    { name: "11 L. Purple", stock: 0 },
    { name: "12 Rose Pink", stock: 12 },
    { name: "13 L. Blue", stock: 5 },
    { name: "16 L. Pink", stock: 0 }
  ];

  function renderStock() {
    const grid = document.getElementById('stockGrid');
    grid.innerHTML = '';
    colorsData.forEach((item) => {
      const card = document.createElement('div');
      card.className = 'color-card';
      const isAvailable = item.stock > 0;
      card.innerHTML = `
        <span class="color-name">${item.name}</span>
        <span class="stock-box ${isAvailable ? 'in-stock' : 'out-of-stock'}">${isAvailable ? item.stock + ' Pkt' : 'NILL'}</span>
      `;
      grid.appendChild(card);
    });
  }

  function shareToBoss() {
    let report = "*Stock Report: D7 Rose Gold Edge Frosted*\\n\\n";
    colorsData.forEach(item => {
      report += (item.stock > 0 ? `✅ *${item.name}*: ${item.stock} Pkt` : `❌ *${item.name}*: NILL`) + '\\n';
    });
    window.open(`https://wa.me/?text=${encodeURIComponent(report)}`, '_blank');
  }

  renderStock();
</script>

</body>
</html>
"""

components.html(html_code, height=800, scrolling=True)
