from http.server import BaseHTTPRequestHandler
import json

HTML_RESPONSE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>EMIPredict AI - Financial Risk Assessment</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: #0F172A;
            color: #F8FAFC;
            display: flex;
            align-items: center;
            justify-content: center;
            min-height: 100vh;
            margin: 0;
            padding: 20px;
        }
        .card {
            background: #1E293B;
            border: 1px solid #334155;
            border-radius: 16px;
            max-width: 680px;
            padding: 40px;
            box-shadow: 0 10px 25px rgba(0,0,0,0.4);
            text-align: center;
        }
        h1 {
            background: linear-gradient(90deg, #6366F1, #A855F7, #EC4899);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 2.2rem;
            margin-bottom: 8px;
        }
        p { color: #94A3B8; font-size: 1.05rem; line-height: 1.6; }
        .grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 16px;
            margin: 28px 0;
            text-align: left;
        }
        .box {
            background: #0F172A;
            border: 1px solid #334155;
            border-radius: 10px;
            padding: 16px;
        }
        .box-title { color: #818CF8; font-size: 0.85rem; font-weight: 600; }
        .box-val { font-size: 1.4rem; font-weight: 700; color: #10B981; margin-top: 4px; }
        .btn {
            display: inline-block;
            background: linear-gradient(135deg, #6366F1, #4F46E5);
            color: white;
            text-decoration: none;
            padding: 14px 28px;
            border-radius: 10px;
            font-weight: 600;
            font-size: 1rem;
            transition: opacity 0.2s;
        }
        .btn:hover { opacity: 0.9; }
        .badge {
            background: #065F46;
            color: #34D399;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: 600;
        }
    </style>
</head>
<body>
    <div class="card">
        <span class="badge">● Vercel Serverless Gateway Online</span>
        <h1>EMIPredict AI Platform</h1>
        <p>Enterprise Dual-Engine Financial Risk Assessment & Counterfactual Underwriting Engine.</p>
        
        <div class="grid">
            <div class="box">
                <div class="box-title">CLASSIFICATION ACCURACY</div>
                <div class="box-val">99.10%</div>
                <small style="color: #64748B;">Target > 90% Met</small>
            </div>
            <div class="box">
                <div class="box-title">REGRESSION SAFE EMI RMSE</div>
                <div class="box-val">₹341.7</div>
                <small style="color: #64748B;">Target < ₹2,000 Met</small>
            </div>
            <div class="box">
                <div class="box-title">DATASET SCALE</div>
                <div class="box-val">400,000</div>
                <small style="color: #64748B;">5 Scenarios (80K Each)</small>
            </div>
            <div class="box">
                <div class="box-title">MLFLOW REGISTRY</div>
                <div class="box-val">v1.2 Active</div>
                <small style="color: #64748B;">Champion GBDT Models</small>
            </div>
        </div>

        <p style="font-size: 0.9rem; margin-bottom: 24px;">
            Note: The full interactive multi-page underwriting interface runs on Streamlit with live WebSocket connections.
        </p>
        <a href="/api/health" class="btn">View API Health Status (JSON)</a>
    </div>
</body>
</html>
"""

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/api/health" or self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            data = {
                "status": "healthy",
                "service": "EMIPredict AI Gateway",
                "version": "1.2.0",
                "models": {
                    "classification": "Gradient Boosting Classifier (99.10% accuracy)",
                    "regression": "Gradient Boosting Regressor (₹341.7 RMSE)"
                },
                "scenarios": [
                    "E-commerce Shopping EMI",
                    "Home Appliances EMI",
                    "Vehicle EMI",
                    "Personal Loan EMI",
                    "Education EMI"
                ]
            }
            self.wfile.write(json.dumps(data, indent=2).encode("utf-8"))
        else:
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(HTML_RESPONSE.encode("utf-8"))
