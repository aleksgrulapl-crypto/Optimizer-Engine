def get_presets(ticker: str):
    emaLenDefault = 200

    stMultiplier = 3.0
    stPeriod = 10
    atrSLmult = 1.4
    atrTPmult = 2.4
    emaLen = emaLenDefault

    if ticker == "NVDA":
        stMultiplier = 1.7
        stPeriod = 8
        atrSLmult = 1.4
        atrTPmult = 4.0
        emaLen = 50
    elif ticker == "MU":
        stMultiplier = 2.6
        stPeriod = 11
        atrSLmult = 2.3
        atrTPmult = 2.3
        emaLen = 185
    elif ticker == "MSFT":
        stMultiplier = 4.0
        stPeriod = 10
        atrSLmult = 1.6
        atrTPmult = 2.8
        emaLen = 250
    elif ticker == "PLTR":
        stMultiplier = 3.4
        stPeriod = 10
        atrSLmult = 1.4
        atrTPmult = 3.1
        emaLen = 160
    elif ticker == "QBTS":
        stMultiplier = 1.8
        stPeriod = 12
        atrSLmult = 1.6
        atrTPmult = 3.0
        emaLen = 180
    elif ticker in ["AAPL","AMD","META","INTC","TSLA","AMZN","SPCX","NFLX",
                    "AVGO","GOOG","WDC","MRVL","STX","AMAT","LRCX","ISRG",
                    "LITE","WMT","CSCO","PLUG"]:
        stMultiplier = 3.0
        stPeriod = 10
        atrSLmult = 1.4
        atrTPmult = 2.4
        emaLen = emaLenDefault
    elif ticker in ["ORCL","UNH","NBIS","BE","LLY"]:
        stMultiplier = 3.0
        stPeriod = 10
        atrSLmult = 1.4
        atrTPmult = 2.4
        emaLen = 200

    return {
        "stMultiplier": stMultiplier,
        "stPeriod": stPeriod,
        "atrSLmult": atrSLmult,
        "atrTPmult": atrTPmult,
        "emaLen": emaLen,
    }
