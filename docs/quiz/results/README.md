# Historia wyników

| Data | Wynik | % | Braki | Link |
|------|-------|---|-------|------|
| 24.04.2026 | 15/20 | 75% | 5, 12, 15, 17 (X); pyt. 18: błędna litera | [wynik](24.04.2026_results.md) |
| 21.04.2026 | 14/20 | 70% | 2, 7, 11, 12, 15, 17 (niedozwolona litera X) | [wynik](21.04.2026_results.md) |
| 20.04.2026 | 18/20 | 90% | 1, 18 (niedozwolona litera X) | [wynik](20.04.2026_results.md) |
| 19.04.2026 | 18/20 | 90% | 9, 19 (niedozwolona litera X) | [wynik](19.04.2026_results.md) |
| 15.04.2026 | 19/20 | 95% | brak (pyt. 7: niedozwolona litera X) | [wynik](15.04.2026_results.md) |

## Statystyki

- **Najlepszy wynik:** 19/20 (95%) — 15.04.2026
- **Najgorszy wynik:** 14/20 (70%) — 21.04.2026
- **Średnia ze wszystkich prób:** 16,8/20 (84%)
- **Ostatnie 5 wyników:** 15/20 (24.04.2026), 14/20 (21.04.2026), 18/20 (20.04.2026), 18/20 (19.04.2026), 19/20 (15.04.2026)

## Najczęstsze obszary do poprawy

Na podstawie wszystkich plików `*_results.md` (łącznie 5 prób), zliczanie tagów w sekcjach szczegółów błędów:

| Kategoria (tag) | Liczba wystąpień błędów |
|-----------------|------------------------|
| [ML] | 16 |
| [scikit-learn] | 1 |

Najmocniejsze wzorce: wpisy `X` zamiast A–D (nadal liczone jak brak), pomyłki na granicy **nadzór / nienadzór / RL** oraz niedoprecyzowana odpowiedź przy pytaniach o **stratę vs koszt** i o **krok SGD (średni koszt, iloczyn cech i błędu)**. Kategoria [scikit-learn] pojawia się przy pytaniach o API `load_iris` i `return_X_y`.

## Notatki

Ranking kategorii nabiera sensu, gdy pojawią się kolejne pliki `*_results.md` i zróżnicowane tagi błędów.

We wcześniejszych wersjach tabeli „Historia” wszystkie niezaliczone pytania z quizu 21.04.2026 to były wyłącznie „X” przy odpowiedziach z bloku str. 70–79 (pyt. 11, 12, 15, 17) oraz definicji straty/kosztu, CV i `return_X_y` (pyt. 2, 7, 15).
