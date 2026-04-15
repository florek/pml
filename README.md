# pml

Repo pod naukę — ML, ćwiczenia z kodem, porządek w projekcie. Spokojnie, krok po kroku.

## Dokumentacja

[W notatkach](docs/lessons/wiedza.md) jest opisane, co jest w repo i co robi lekcja z perceptronem — napisane po ludzku, bez przesady.

Quiz do utrwalenia materiału z notatek: [pytania](docs/quiz/questions/15.04.2026_questions.md), [moje odpowiedzi (szablon)](docs/quiz/my_answers/15.04.2026_my_answers.md), [klucz](docs/quiz/answers/15.04.2026_answers.md).

## Start w kilku krokach

```text
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

W Cursorze / VS Code wybierz interpreter z `venv` (albo zostaw domyślny z `.vscode/settings.json`), żeby Pylance/basedpyright widział zainstalowane paczki (`pandas`, `matplotlib`).

Lekcja z kodem: `src/p52/main.py` (dane: `src/p52/iris.data` — lokalna kopia UCI, bez pobierania z internetu; wykresy: `matplotlib` z `requirements.txt`).
