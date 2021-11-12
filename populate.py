import os, requests, json

base_url = 'https://cloud.iexapis.com/v1'
sandbox_url = 'https://sandbox.iexapis.com/v1'

# token = os.environ.get('IEX_TOKEN')
token = 'Tsk_3339c0f8d2e34f4db867c257fcffa153'
params = {
    'token': token,
    'types': 'chart',
    'range': '25y'
}
symbols = 'SPY,DIA,DAX,FEZ,AIA,EEM,MDY,IWM,FXE,UUP,GLD,HYG,IEF,TLT,VXX,VIXM,TAN,XLE,XLF,XLI,XLP,DBC,DBB'.split(',')

## Common Indices
# SPY - SP500
# DIA - DJIA
# DAX - DAX
# FEZ - STOXX50
# AIA - ASIA50

## By Cap Size
# EEM - Emerging markets
# MDY - Mid Cap
# IWM - Smallcap

## FX
# FXE - Euro Index
# UUP - Dollar Index
# GLD - Gold

## Bonds
# HYG - High Yield
# IEF - US10YR (inverted)
# TLT - US20YR (inverted?)

## Other Indicators
# VXX - VIX Short Term
# VIXM - VIX Mid Term

## Sectors
# TAN - Green Energy
# XLE - Brown Energy
# XLF - Finance
# XLI - Industrial
# XLP - Consumer Staples
# DBC - Commodities
# DBB - Base Metals

with open('./data.json', 'w') as f:

    symbols_count = len(symbols)

    f.write('{')

    for i, sym in enumerate(symbols):
        print(f'{sym}: symbol {i + 1} out of {symbols_count}...')

        res = requests.get(sandbox_url + f'/stock/{sym}/chart', params=params)
        res.raise_for_status()
        raw_data = res.json()

        less_data = list(map(
            lambda item: {
                "open": str(item['open']),
                "close": str(item['close']),
                "high": str(item['high']),
                "low": str(item['low']),
                "volume": str(item['volume']),
                "date": item['date'],
                "changePercent": str(item['changePercent']),
                "label": item['label']
            }, raw_data))

        f.write(f'"{sym}": {json.dumps(less_data, indent=4)}')
        if (i < symbols_count - 1):
            f.write(',')

    f.write('}')
