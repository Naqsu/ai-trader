# Conservative Multi-Agent AI Trading Bot

Szkielet projektu pod profesjonalnego, konserwatywnego bota tradingowego dla paper tradingu futures, z późniejszą ścieżką do prop firm challenge/combine oraz ewentualnie IBKR.

## Główne założenia

- bezpieczeństwo > zwrot,
- 0.5-1.0% ryzyka na transakcję,
- minimalny Risk/Reward 1.5-2.0,
- konserwatywna adaptacja modeli i wag strategii,
- `RiskGuardianAgent` ma twarde prawo weta,
- każda transakcja przechodzi double-check przed wykonaniem,
- najpierw stabilność, potem optymalizacja.

## Główna klasa

- `PaperFuturesMultiAgentBot`

## Rynki i dane

Startowy zakres:

- NAS100 Historical Price Data,
- multi-timeframe OHLCV,
- `1m`, `5m`, `15m`, `30m`, `1h`, `4h`, `1d`, `1w`, `1month`.

Docelowy zakres:

- NQ / MNQ futures jako główny fokus,
- rozszerzenia pod `GC`, `CL`, `SI`, `HG`, `NG`,
- przyszłe integracje danych i execution pod Databento, CME, Rithmic, Tradovate i IBKR.

## Architektura katalogów

```text
bot/
  core/
  data/
  indicators/
  strategies/
  risk/
  execution/
  memory/
  ai/
  backtesting/
  reporting/
  utils/
tests/
scripts/
storage/
  raw_data/
  processed_data/
  models/
  logs/
  reports/
  state/
```

## Przepływ decyzyjny

```text
DataAgent
-> IndicatorAgent
-> MarketRegimeAI
-> StrategyAgent
-> SetupQualityAI
-> EpisodicMarketMemory
-> MistakeLearner
-> DoubleCheckAgent
-> RiskGuardianAgent
-> RiskManager
-> ExecutionAgent
-> TradeLogger / DecisionLogger
-> MetaSupervisorAI / DailyReport
```

## Odpowiedzialności komponentów

- `DataAgent`: pobieranie, walidacja i przygotowanie danych.
- `IndicatorAgent`: liczenie VWAP, EMA, RSI, ATR, momentum i volatility regime.
- `StrategyAgent`: obsługa strategii `trend_vwap`, `reversion_rsi`, `breakout_atr`.
- `MarketRegimeAI`: klasyfikacja reżimu rynku.
- `SetupQualityAI`: ocena jakości setupu.
- `RiskFilterAI`: pomocnicza analiza ryzyka dla `RiskGuardianAgent`.
- `RiskGuardianAgent`: finalne, twarde weto.
- `DoubleCheckAgent`: końcowa weryfikacja sygnału przed wejściem.
- `RiskManager`: sizing i limity ryzyka.
- `ExecutionAgent`: paper execution z prowizją i slippage.
- `EpisodicMarketMemory`: zapis kontekstu rynkowego i decyzji.
- `MistakeLearner`: analiza powtarzalnych strat.
- `StrategyWeightAI`: ostrożna aktualizacja wag strategii.
- `MetaSupervisorAI`: raportowanie i nadzór bez prawa wykonania transakcji.
- `TradeLogger` i `DecisionLogger`: krytyczne logowanie do uczenia i audytu.

## Status

Ten etap zawiera wyłącznie profesjonalny szkielet projektu:

- struktura katalogów,
- minimalne pliki modułów,
- placeholdery klas,
- pliki konfiguracyjne,
- skrypt automatycznie budujący strukturę.

Bez implementacji logiki tradingowej.

## Uruchomienie generatora struktury

```bash
python scripts/build_project_structure.py
```
