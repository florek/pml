# Notatki z kursu — od czego zaczynamy

Luźny styl, ale treść ma być ściągą: definicje, „dlaczego tak”, typowe pułapki. Sama wiedza — bez mapowania na konkretne pliki projektu.

## Po co ćwiczymy ML w Pythonie

Uczymy się łączyć matematykę klasyfikacji z kodem: dane jako wektory cech, model jako prosta funkcja decyzyjna, trening jako powtarzalna aktualizacja parametrów. Pierwszy krok to często **perceptron** — najprostszy klasyfikator liniowy, który dobrze tłumaczy ideę **granicy decyzji** i **uczenia online** (aktualizacja po jednej próbce). W praktyce ten sam schemat „wejście liniowe → decyzja” powtarza się w większych modelach: pojedynczy perceptron to jak jeden neuron liniowy z progiem; sieci składają się z wielu takich bloków (zwykle z **nieliniowością** między warstwami), a współczesne modele sekwencyjne też na tym się opierają — tylko w skali i z inną architekturą.

## Uczenie maszynowe — trzy główne nurty

**Uczenie nadzorowane** trenuje model na **oznakowanych** danych: znamy pożądane wyjścia (etykiety lub wartości docelowe). Stąd m.in. **klasyfikacja** (etykiety dyskretne, np. spam / nie spam, litery alfabetu) i **regresja** (wartości ciągłe, np. przewidywana ocena). Przykład klasyfikacji wieloklasowej: odręczne litery — jeśli w danych uczących nie ma cyfr, model nie nagle „zna” cyfr na wyjściu.

**Uczenie przez wzmacnianie** buduje agenta poprawiającego działanie przez interakcję ze środowiskiem; sprzężenie zwrotne to zwykle **nagroda** (nie „prawdziwa etykieta” jak w nadzorze), czasem opóźniona do końca epizodu (np. partia szachów).

**Uczenie nienadzorowane** pracuje na danych **bez znanych etykiet** (lub struktura jest niejasna). Typowe zadania: **klasteryzacja** (grupowanie podobnych obiektów bez z góry nadanych grup), **redukcja wymiarowości** (kompresja cech, odszumianie, czasem też wizualizacja przez rzut na małą liczbę wymiarów).

## Terminologia i notacja

**Cechy** to zwykle zmienne wejściowe (kolumny w tabeli); **zmienna celu** to to, co chcemy przewidzieć (etykieta klasy, wartość ciągła). W macierzy cech **wiersze** to przykłady, **kolumny** to cechy — konwencja wygodna do zapisu wektorowego.

**Funkcja straty** bywa rozróżniana od **funkcji kosztu**: strata często liczy się **dla pojedynczego** punktu, koszt **agreguje** (średnia, suma) straty po całym zbiorze.

**Trening** to dopasowanie parametrów modelu (w modelach parametrycznych zbliżone do **estymacji** parametrów z danych).

## Strategia budowy systemu ML

**Wstępne przetwarzanie** jest prawie zawsze konieczne: surowe dane rzadko pasują idealnie do algorytmu. Przykłady: wyciąganie sensownych cech z bogatszej postaci obserwacji, **skalowanie** cech do wspólnej skali (np. zakres lub standaryzacja ze średnią 0 i wariancją 1), redukcja **nadmiarowych** cech silnie skorelowanych — czasem przez redukcję wymiarowości.

**Podział na zbiór uczący i testowy** (często losowy): uczymy i stroimy na treningowym, **test** trzymamy na koniec do **ostatecznej** oceny uogólnienia. Parametry kroków takich jak skalowanie wyznacza się **tylko na treningu**, potem ta sama transformacja idzie na test i na nowe dane — inaczej metryka może być **zbyt optymistyczna**.

**Twierdzenie „no free lunch”** w praktyce znaczy: nie ma jednego najlepszego algorytmu dla wszystkich problemów — trzeba dopasować założenia do zadania i **porównać** modele.

**Sprawdzian krzyżowy** dzieli część danych uczących na podzbiory uczący i walidacyjny, żeby szacować uogólnienie **bez** podglądania zestawu testowego przy wyborze modelu.

**Hiperparametry** to ustawienia **nie** wynikające bezpośrednio z dopasowania wag z próbek (np. liczba epok, współczynnik uczenia w perceptronie) — regulują zachowanie algorytmu.

**Dokładność klasyfikacji** to częsty wskaźnik: stosunek poprawnych predykcji do wszystkich instancji (w zależności od definicji zadania).

## Środowisko i narzędzia

**Wirtualne środowisko** izoluje wersje bibliotek od systemu i od innych projektów. Aktywujesz je lokalnie, instalujesz paczki tylko „do środka” tego katalogu — mniej konfliktów, powtarzalne uruchomienia u Ciebie i u innych, jeśli współdzielicie listę zależności.

**Plik zależności** to zamrożona lista nazw pakietów i często minimalnych wersji. Zamiast pamiętać „zainstaluj to i tamto”, robisz jedną komendę instalacji z listy — standard w małych i większych projektach.

**Ignorowanie katalogu środowiska w kontroli wersji** ma sens, bo w środku są tysiące plików binarnych i środowiskowych; repozytorium trzyma **przepis** (lista paczek), a nie **gotową kopię** zainstalowanego świata. Klonujesz projekt, odtwarzasz wirtualne środowisko u siebie — lekko i czytelnie.

**Analiza statyczna w edytorze** (podkreślenia importów) patrzy na **ten interpreter Pythona**, który jest wybrany w IDE. Jeśli paczki są zainstalowane tylko w wirtualnym środowisku, a edytor wskazuje na inny Python, zobaczysz fałszywe alarmy — rozwiązanie to ten sam interpreter, w którym faktycznie instalujesz zależności.

**Python** w analizie danych często łączy się z bibliotekami z szybkimi rdzeniami w C/Fortran (**NumPy**, **SciPy**). Do uczenia klasycznego ML wygodna jest **scikit-learn**; **pandas** upraszcza pracę z tabelami; **Matplotlib** służy wizualizacji. **TensorFlow** pojawia się w kursie przy uczeniu głębokim. Instalacja dodatkowych pakietów: standardowo **pip** (`pip install`, `pip install … --upgrade`); alternatywnie dystrybucja **Anaconda** z menedżerem **conda** (`conda install`, `conda update`).

## Dane tabelaryczne i wczytywanie

**Zbiór iris** ma 150 przykładów, cztery cechy numeryczne i etykietę gatunku. Na początku często bierze się **dwie klasy** i **dwie cechy**, żeby wizualizacja była w 2D.

**Wczytanie z dysku zamiast z adresu URL** ma sens przy ćwiczeniu: odczyt przez bibliotekę z sieci wymaga działającego DNS i połączenia; błąd typu „nie udało się rozwiązać nazwy hosta” oznacza problem sieciowy, nie matematyczny modelu.

**Mapowanie etykiet tekstowych na liczby** dla dwóch klas w perceptronie: jedna klasa jako wartość ujemna, druga jako dodatnia (np. `-1` i `1`) — zgodnie z regułą uczenia.

**Wybór podzbioru wierszy i kolumn** w ramce danych: pierwsze sto wierszy na problem binarny dwóch gatunków, konkretne kolumny jako współrzędne w przestrzeni cech (np. długość działki kielicha i długość płatka).

## NumPy w roli „silnika” pod model

**Tablice** reprezentują wektory cech i wagi; operacje typu iloczyn skalarny robi się funkcjami zoptymalizowanymi pod CPU, zamiast ręcznych pętli po elementach w czystym Pythonie. **Wektoryzacja** pozwala wykonywać operacje na całych tablicach; pod spodem często wykorzystuje się SIMD oraz biblioteki algebry liniowej (**BLAS**, **LAPACK**).

**Generator z ustalonym ziarnem** daje powtarzalny start losowych wag: ten sam seed → te same liczby przy kolejnym uruchomieniu. Przydaje się przy debugowaniu i porównywaniu eksperymentów.

**Małe wartości startowe wag** z rozkładu normalnego (blisko zera, niska wariancja) to typowy trik: unikasz zbyt dużych aktywacji na wejściu i „płaskiego” startu treningu. **Same zera** jako inicjalizacja wag sprawiają, że przy mnożeniu przez skalar zmienia się tylko skala wektora, **nie kierunek** — współczynnik uczenia nie jest w stanie „wybrać” sensownego kierunku rozdzielenia klas; stąd **niezerowe** małe startowe wagi.

**Indeksowanie** tablic dwuwymiarowych: pierwszy indeks to wiersz, drugi kolumna.

**Kombinacja liniowa** cech i wag plus bias to dokładnie to, co liczy się przed progiem: suma ważonych wejść i przesunięcie. **Próg** można zapisać jako **obciążenie jednostkowe** (bias): sztuczna cecha równa 1 i odpowiadająca jej waga `w₀`, co daje zwarty zapis **iloczynu skalarnego** `wᵀx` z rozszerzonym wektorem cech.

**Prog warunkowy na całych tablicach** (`wartość jeśli warunek, w przeciwnym razie inna`) pozwala w jednym kroku przypisać klasę tam, gdzie wartość przed progiem jest nienegatywna, i drugą klasę w przeciwnym razie — wygodne przy wektorze predykcji i przy **kodowaniu etykiet** z nazw klas.

**Siatka punktów w 2D** do rysowania granicy: dwie osie zakresów, **iloczyn kartezjański** zakresów przez funkcję generującą siatkę, spłaszczenie do listy punktów, predykcja klasyfikatora dla każdego punktu, **przekształcenie z powrotem** do kształtu siatki — pod **wypełnione kontury** kolorami klas.

## Matplotlib — wizualizacja w ćwiczeniu

**Wykres rozrzutu** dwóch cech: osobne serie dla klas (kolor, marker), oś X i Y z opisem fizycznym cechy.

**Wykres błędu w czasie treningu** oś pozioma to numer epoki (od jedynki), pionowa to liczba **niezerowych korekt** w epoce — sygnał, czy model jeszcze się poprawia.

**Regiony decyzji:** tło jako wypełnione kontury z lekką przezroczystością, na wierzchu te same próbki co wcześniej; **mapa kolorów** ograniczona do tylu kolorów, ile jest **unikalnych etykiet** po treningu; `ListedColormap` łączy listę kolorów z liczbą klas.

**Backend bez interakcji** (np. rasteryzacja do pliku): wywołanie pokazujące okno może dać ostrzeżenie lub nic nie pokazać — to kwestia środowiska uruchomienia, nie samego modelu.

## Neuron i perceptron — kontekst historyczny i formalny

Model **McCullocha-Pittsa** (1943) to uproszczony neuron jako bramka logiczna z binarnym wyjściem. **Rosenblatt** zaproponował **regułę uczenia perceptronu** (1957): wagi mnożą wejścia, decyzja zależy od przekroczenia progu — nadaje się do klasyfikacji z przypisaniem punktów do klas.

Dla **klasyfikacji binarnej** wygodne są etykiety **1** i **-1**. **Całkowite pobudzenie** to iloczyn skalarny wag i cech (plus bias w zapisie rozszerzonym). **Funkcja decyzyjna** perceptronu to **skok jednostkowy**: po jednej stronie progu jedna klasa, po drugiej druga.

## Perceptron — mechanika

**Etykiety** w tym wariancie to dwie klasy wyrażone liczbowo: dodatnia i ujemna (np. `1` i `-1`). Taki wybór upraszcza regułę aktualizacji: znak błędu od razu mówi, w którą stronę pchnąć wagę.

**Bias** można trzymać jako osobny parametr albo **dokleić sztuczną cechę** i traktować go jak kolejną wagę — w kodzie często widać wektor wag dłuższy o jeden element: pierwszy to próg, reszta to wagi cech.

**Reguła uczenia:** aktualizacja wagi jest proporcjonalna do **współczynnika uczenia η** (często 0–1), różnicy **prawdziwa etykieta minus przewidywana** oraz odpowiednio do **cechy** (dla biasu „cecha” to jedynka). Wszystkie wagi aktualizuje się **w tej samej** iteracji po próbce, z jednym obliczeniem błędu — nie przelicza się pośrednich predykcji po każdej pojedynczej wadze w tej samej próbce w podstawowym wariancie z książki.

**Pętla zewnętrzna — epoki:** jedna epoka to przejście po wszystkich próbkach (w ustalonej kolejności w najprostszym wariancie). Parametr liczby epok ogranicza czas treningu; przy klasach **nieseparowalnych liniowo** błąd nie musi spaść do zera — wtedy sens ma **górne ograniczenie epok** (i ewentualnie próg tolerancji), żeby algorytm nie kręcił się w nieskończoność.

**Predykcja przed progiem:** jeśli **suma ważona + bias** jest większa lub równa zero, przypisujesz jedną klasę; w przeciwnym razie drugą. Granica to hiperpłaszczyzna w przestrzeni cech.

**Zbieżność** reguły perceptronu jest gwarantowana w sensie klasycznym, gdy **obie klasy są liniowo rozdzielne** i współczynnik uczenia jest **wystarczająco mały**. Gdy liniowej granicy nie da się idealnie postawić, wagi mogą się aktualizować bez końca — stąd limit epok.

**Licznik „popsutych” aktualizacji w epoku:** zliczasz, ile razy w danej epoce krok uczenia był niezerowy (czyli model jeszcze poprawiał się na próbce). To nie jest dokładność na zbiorze testowym, ale **sygnał, czy trening jeszcze coś zmienia** — przydatny przy szybkim podglądzie zbieżności.

## Interfejs obiektowy klasyfikatora

Typowy wzorzec: metoda **dopasowania** uczącego zbioru oraz osobna metoda **prognozy**. Atrybuty powstające dopiero po treningu (np. wektor wag) bywa oznaczany **końcowym podkreślnikiem** w konwencji scikit-learn (`w_`, `errors_`), żeby odróżnić je od argumentów konstruktora.

## Klasyfikacja wieloklasowa a perceptron binarny

Perceptron w podstawowej postaci jest **binarny**. Rozszerzenie na wiele klas: strategia **jeden kontra reszta** (**OvR** / czasem **OvA**): dla każdej klasy jeden klasyfikator z tą klasą jako pozytywną a resztą jako negatywną; przy perceptronach wybór klasy można powiązać z **największą wartością bezwzględną** całkowitego pobudzenia między modelami.

## Od perceptronu do Adaline (zapowiedź)

**Adaline** (adaptacyjny neuron liniowy) aktualizuje wagi na podstawie **ciągłej** odpowiedzi liniowej (funkcja aktywacji tożsamościowa na pobudzeniu), a **dopiero** ostateczna decyzja może używać progu podobnego do skoku. **Perceptron** porównuje etykiety z **już** przyciętymi predykcjami dyskretnymi przy aktualizacji wag. Ta różnica wiąże Adaline z ideą **minimalizacji ciągłej funkcji kosztu** (np. błędów kwadratowych), co prowadzi dalej w stronę gradientu i modeli jak regresja logistyczna czy SVM.

## Pułapki i dobre nawyki

**Separowalność liniowa:** klasyczny perceptron ma gwarancje sensowne tylko wtedy, gdy klasy da się oddzielić płaszczyzną. Przy nakładających się chmurach punktów błąd treningowy może oscylować — wtedy inne modele lub cechy są konieczne.

**Współczynnik uczenia:** za duży — oscylacje i niestabilność; za mały — powolny postęp. W praktyce zwykle eksperymentalnie lub z rozszerzeniami (np. później: regularyzacja, inne reguły optymalizacji).

**Powtarzalność:** ustalone ziarno losowe i ta sama kolejność epok dają powtarzalny przebieg na tym samym kodzie i danych — ułatwia porównanie dwóch wersji algorytmu.

**Spójność środowiska:** jeden Python do instalacji paczek i do uruchamiania skryptów; ten sam interpreter w IDE co w terminalu — mniej chaosu z importami i wersjami.

**Dane lokalne vs sieć:** ćwicząc wczytywanie tabeli, trzymaj kopię pliku u siebie, jeśli chcesz uniknąć zależności od połączenia i DNS w momencie nauki.
