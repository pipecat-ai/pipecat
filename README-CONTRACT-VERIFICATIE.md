# Contract Verificatie Systeem

Dit systeem biedt automatische contractverificatie via telefoongesprekken met behulp van Twilio en AI.

## Functionaliteiten

- Automatisch telefoongesprek starten vanuit Salesdock webhook
- Verificatie van contractgegevens in het Nederlands
- Opname van het gesprek
- Opslag van verificatieresultaten in Google Sheets
- Notificatie naar backoffice na succesvolle verificatie
- Optie om door te verbinden naar een medewerker (toets 1)

## Vereisten

- Python 3.8+
- Pipecat AI framework
- Twilio account
- OpenAI API key
- Deepgram API key (voor spraakherkenning)
- Cartesia API key (voor text-to-speech)
- Google Sheets API credentials

## Installatie

1. Maak een virtuele omgeving en activeer deze:

```bash
python3 -m venv venv
source venv/bin/activate
```

2. Installeer de benodigde pakketten:

```bash
pip install "pipecat-ai[daily,deepgram,cartesia,openai,silero]"
pip install twilio gspread oauth2client python-dotenv
```

3. Maak een `.env` bestand aan op basis van het template en vul de vereiste API keys in:

```bash
cp .env.template .env
```

## Configuratie

### Twilio configuratie

1. Log in op je Twilio dashboard
2. Maak een nieuw telefoonnummer aan of gebruik een bestaand nummer
3. Configureer de webhook URL voor inkomende gesprekken
4. Zet je Account SID, Auth Token en telefoonnummer in het `.env` bestand

### Google Sheets configuratie

1. Maak een nieuw Google Cloud project aan
2. Schakel de Google Sheets API in
3. Maak een service account aan en download de credentials als JSON
4. Deel je Google Sheet met het e-mailadres van het service account
5. Zet het pad naar het credentials bestand en de naam van de sheet in het `.env` bestand

### Salesdock integratie

1. Configureer een webhook in Salesdock die activeert bij nieuwe contracten
2. Stel in dat de webhook de contractgegevens in JSON-formaat naar jouw API endpoint stuurt

## Gebruik

1. Start de API server:

```bash
python api_server.py
```

2. De server zal luisteren naar webhook calls van Salesdock
3. Wanneer een nieuw contract wordt aangemaakt, belt het systeem automatisch de opgegeven telefoonnummer
4. Het systeem stelt de verificatievragen en verwerkt de reacties
5. Na succesvolle verificatie worden de gegevens opgeslagen in Google Sheets

## Testing

Voor testdoeleinden kun je het script rechtstreeks uitvoeren met voorbeeldgegevens:

```bash
python contract_verification_bot.py
```

## Belangrijke opmerkingen

- Voor productiebouw installeer je best deze applicatie op een server die 24/7 online is
- Overweeg om monitoring en logging toe te voegen voor productiegebruik
- Bekijk de privacy- en GDPR-implicaties van gespreksopnames en dataopslag
- Test grondig voordat je het systeem in productie neemt 