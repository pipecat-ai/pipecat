# 📞 Phone Chatbot with Real-Time Company Awareness  
**(Twilio + Daily SIP + Tavily + FastAPI)**

This project creates a phone chatbot that:

- Handles real inbound voice calls using **Twilio**
- Connects to a SIP-enabled bot via **Daily.co**
- Detects **company names** from the caller's voice
- Fetches **real-time background info** using **Tavily**
- Uses that info to make the conversation **more personalized**
- Ends the call after 3 missed (silent) responses
- Logs a **post-call summary** including duration, silence, and company name

---

## 🚀 Features

✅ Inbound voice call handling  
✅ Real-time company lookup  
✅ SIP bot integration via Daily  
✅ Context-aware follow-up prompts  
✅ Silent caller detection  
✅ Post-call logging

---

## 🧠 How It Works

1. 📞 **Caller dials your Twilio number**
2. 🎤 Bot says: _“Tell me the name of the company you're interested in.”_
3. 🧠 If a company is mentioned (e.g., "OpenAI"):
   - The bot uses **Tavily** to find relevant info
   - Replies: _“I found this about OpenAI: ... What would you like to ask about that?”_
4. 🤐 If the user says nothing (3x), the call ends
5. 🧾 Server logs a summary (call duration, company, missed prompts)

---

## 🛠 Setup Guide

### 1. Clone the Repo

```bash
git clone https://github.com/MahdisEsm/pipecat
cd examples/phone-chatbot-extended
```

---

### 2. Install Requirements

```bash
pip install -r requirements.txt
pip install tavily-python
```

---

### 3. Create `.env` File

Create a `.env` file in the root with:

```env
TWILIO_ACCOUNT_SID=your_twilio_sid
TWILIO_AUTH_TOKEN=your_twilio_auth_token
TWILIO_PHONE_NUMBER=+1234567890
DAILY_API_KEY=your_daily_api_key
TAVILY_API_KEY=your_tavily_api_key
PORT=8000
```

---

### 4. Run the Server

```bash
python server.py
```

It starts at:  
http://localhost:8000

---

### 5. Expose with ngrok

```bash
ngrok http 8000
```

Copy the HTTPS forwarding URL (e.g. `https://abcd1234.ngrok.io`)

---

### 6. Set Up Twilio Webhooks

In [Twilio Console → Phone Numbers](https://console.twilio.com/):

- Under **Voice → A CALL COMES IN**, set to:
  ```
  Webhook POST → https://your-ngrok-url.ngrok.io/call
  ```

- (Optional) Under **Status Callback**, set:
  ```
  https://your-ngrok-url.ngrok.io/status-callback
  ```

---

## 🧪 Testing

- **Say a company name** during the call (e.g., "Tell me about Microsoft")
- **Don’t respond** to test silent user handling
- **Watch the terminal logs** to view the call summary:
  ```
  📞 Call Summary:
  - Duration: 45s
  - Missed Prompts: 1
  - Company Info Used: "Microsoft is a tech company based in Redmond..."
  ```

---

## 📂 Project Structure

```
phone-chatbot-daily-twilio-sip/
├── server.py              # FastAPI webhook server
├── bot.py                 # SIP-aware chatbot process
├── utils/
│   └── daily_helpers.py   # Creates Daily SIP rooms
├── .env.example           # Sample environment configuration
├── README.md              # You're here!
└── requirements.txt       # Python dependencies
```

---

## 🔧 Requirements

- Python 3.8+
- Twilio account (trial is OK)
- Daily.co API key (https://www.daily.co/)
- Tavily API key (https://www.tavily.com/)
- Ngrok (https://ngrok.com/)

---

## 💡 Example Use Case

A user calls and says:  
_“Hi, I’d like to learn more about Stripe.”_

Bot responds with:  
_“I found this about Stripe: Stripe builds financial infrastructure for the internet. What would you like to ask about them?”_

---

## 🛟 Need Help?

- Make sure your ngrok tunnel is active
- Double-check Twilio webhook is set to `/call`
- Use logs in `server.py` to debug missing company matches

---

## 🧾 License

MIT
