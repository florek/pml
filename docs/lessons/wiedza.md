# Notatki z kursu — od czego zaczynamy

Luźny styl, ale treść ma być ściągą: definicje, „dlaczego tak”, typowe pułapki. Bez mapowania „co w którym pliku” — sama wiedza.

## Po co ćwiczymy ML w Pythonie

Uczymy się łączyć matematykę klasyfikacji z kodem: dane jako wektory cech, model jako prosta funkcja decyzyjna, trening jako powtarzalna aktualizacja parametrów. Pierwszy krok to często **perceptron** — najprostszy klasyfikator liniowy, który dobrze tłumaczy ideę **granicy decyzji** i **uczenia online** (aktualizacja po jednej próbce).

## Środowisko i narzędzia

**Wirtualne środowisko** izoluje wersje bibliotek od systemu i od innych projektów. Aktywujesz je lokalnie, instalujesz paczki tylko „do środka” tego katalogu — mniej konfliktów, powtarzalne uruchomienia u Ciebie i u innych, jeśli współdzielicie listę zależności.

**Plik zależności** to zamrożona lista nazw pakietów i często minimalnych wersji. Zamiast pamiętać „zainstaluj to i tamto”, robisz jedną komendę instalacji z listy — standard w małych i większych projektach.

**Ignorowanie katalogu środowiska w kontroli wersji** ma sens, bo w środku są tysiące plików binarnych i środowiskowych; repozytorium trzyma **przepis** (lista paczek), a nie **gotową kopię** zainstalowanego świata. Klonujesz repo, odtwarzasz venv u siebie — lekko i czytelnie.

## NumPy w roli „silnika” pod model

**Tablice** reprezentują wektory cech i wagi; operacje typu iloczyn skalarny robi się funkcjami zoptymalizowanymi pod CPU, zamiast ręcznych pętli po elementach w czystym Pythonie.

**Generator z ustalonym ziarnem** daje powtarzalny start losowych wag: ten sam seed → te same liczby przy kolejnym uruchomieniu. Przydaje się przy debugowaniu i porównywaniu eksperymentów.

**Małe wartości startowe wag** z rozkładu normalnego (blisko zera, niska wariancja) to typowy trik: unikasz zbyt dużych aktywacji na wejściu i „płaskiego” startu treningu.

**Kombinacja liniowa** cech i wag plus bias to dokładnie to, co liczy się przed progiem: suma ważonych wejść i przesunięcie. To serce warstwy liniowej w wielu modelach.

**Prog warunkowy na całych tablicach** pozwala w jednym kroku przypisać klasę tam, gdzie wartość przed progiem jest nienegatywna, i drugą klasę w przeciwnym razie — wygodne przy wektorze predykcji.

## Perceptron — mechanika

**Etykiety** w tym wariancie to dwie klasy wyrażone liczbowo: dodatnia i ujemna (np. `1` i `-1`). Taki wybór upraszcza regułę aktualizacji: znak błędu od razu mówi, w którą stronę pchnąć wagę.

**Bias** można trzymać jako osobny parametr albo **dokleić sztuczną cechę** i traktować go jak kolejną wagę — w kodzie często widać wektor wag dłuższy o jeden element: pierwszy to próg, reszta to wagi cech.

**Reguła uczenia:** dla każdej próbki liczysz różnicę między prawdziwą etykietą a aktualną predykcją, mnożysz przez **współczynnik uczenia** (często oznaczany grecką etą) i dodajesz do wag składową proporcjonalną do wejścia; bias aktualizujesz tak, jakby wejście „stałej jedynki” było zawsze obecne — czyli sam skalar tej aktualizacji ląduje na progu.

**Pętla zewnętrzna — epoki:** jedna epoka to przejście po wszystkich próbkach (w ustalonej kolejności w najprostszym wariancie). Parametr liczby epok ogranicza czas treningu; za mało — model nie zdąży się nauczyć, za dużo — przy danych nieseparowalnych liniowo i tak nie „zamazuje” błędu do zera, a koszt obliczeń rośnie.

**Predykcja przed progiem:** jeśli **suma ważona + bias** jest większa lub równa zero, przypisujesz jedną klasę; w przeciwnym razie drugą. Granica to hiperpłaszczyzna w przestrzeni cech.

**Licznik „popsutych” aktualizacji w epoku:** zliczasz, ile razy w danej epoce krok uczenia był niezerowy (czyli model jeszcze poprawiał się na próbce). To nie jest dokładność na zbiorze testowym, ale **sygnał, czy trening jeszcze coś zmienia** — przydatny przy szybkim podglądzie zbieżności.

## Pułapki i dobre nawyki

**Separowalność liniowa:** klasyczny perceptron ma gwarancie sensowne tylko wtedy, gdy klasy da się oddzielić płaszczyzną. Przy nakładających się chmurach punktów błąd treningowy może oscylować — wtedy inne modele lub cechy są konieczne.

**Współczynnik uczenia:** za duży — oscylacje i niestabilność; za mały — powolny postęp. W praktyce zwykle eksperymentalnie lub z rozszerzeniami (np. później: regularyzacja, inne reguły optymalizacji).

**Powtarzalność:** ustalone ziarno losowe i ta sama kolejność epok dają powtarzalny przebieg na tym samym kodzie i danych — ułatwia porównanie dwóch wersji algorytmu.

**Izolacja środowiska + lista paczek** to minimum higieny: nie mieszasz bibliotek z pięciu projektów w jednym globalnym Pythonie i nie wrzucasz gigabajtów `site-packages` do Gita.

W `src/p52/iris.data` leży lokalna kopia zbioru iris z UCI — `main.py` czyta ten plik z dysku. Dzięki temu unikasz błędu sieci/DNS przy `pd.read_csv` z adresu URL (np. `URLError: getaddrinfo failed`, czyli brak rozwiązania nazwy hosta albo brak połączenia).

Wykresy w tej lekcji idą przez **matplotlib** (jest w `requirements.txt`). Żeby edytor nie podkreślał importów na czerwono, trzymaj aktywne **venv** z zainstalowanymi paczkami i wybierz ten interpreter w IDE (w repo jest podpowiedź w `.vscode/settings.json` i `pyrightconfig.json`).

---

*Ściąga spięta z tym, co realnie ćwiczysz w repozytorium: perceptron z NumPy, venv, zależności pip, sensowne ignorowanie środowiska w Git.*
