1. Title: Bach Chorales Harmony

2. Source Information
   -- Creators: Daniele P. Radicioni and Roberto Esposito
   -- Donor: Daniele P. Radicioni (radicion@di.unito.it) and Roberto Esposito (esposito@di.unito.it)
   -- Date: May, 2014

3. Past Usage:
        1. D. P. Radicioni and R. Esposito. Advances in Music Information Retrieval,
        chapter BREVE: an HMPerceptron-Based Chord Recognition System. Studies
        in Computational Intelligence, Zbigniew W. Ras and Alicja Wieczorkowska
        (Editors), Springer, 2010.
        2. Esposito, R. and Radicioni, D. P., CarpeDiem: Optimizing the Viterbi
          Algorithm and Applications to Supervised Sequential Learning, Journal
          of Machine Learning Research, 10(Aug):1851-1880, 2009.

       -- Results:
          -- prediction of the labels of the chord resonating for each
             event (accuracy 80.06%)

  - Predicted attribute: chord label

4. Relevant Information:
   -- Abstract
      Given a musical flow, the task of music harmony analysis consists in associating 
      a label to each time point. Such labels reveal the underlying harmony by indicating
      a fundamental note (root) and a mode, using chord names such as ‘C minor’.

      The music analysis task can be naturally represented as a supervised sequential learning
      problem. In fact, by considering only the currently resonating pitch classes, one
      would hardly produce reasonable analyses. Experimental evidences about human
      cognition reveal that in order to disambiguate unclear cases, composers and listeners
      refer to chord transitions as well: in these cases, context plays a
      fundamental role, and contextual cues can be useful to the analysis.

      The data set is composed of 60 chorales (5665 events) by J.S. Bach (1675-1750).
      Each event of each chorale is labelled using 1 among 101 chord labels.

   -- Pitch classes information has been extracted from MIDI sources downloaded
      from (JSB Chorales)[http://www.jsbchorales.net]. Meter information has
      been computed through the Meter program which is part of the Melisma
      music analyser (Melisma)[http://www.link.cs.cmu.edu/music-analysis/].
      Chord labels have been manually annotated by a human expert.

5. Number of Instances: 60 sequences, 5665 events

6. Number of Attributes: 17 (sequence name, event number, notes presence (x12),
                              chord)

7. Attribute Information:
   1. Choral ID: corresponding to the file names from (Bach Central)[http://www.bachcentral.com].
   2. Event number: index (starting from 1) of the event inside the chorale.
   3-14. Pitch classes: YES/NO depending on whether a given pitch is present.
      Pitch classes/attribute correspondence is as follows:
        C       -> 3
        C#/Db   -> 4
        D       -> 5
        ...
        B       -> 14
   15. Bass: Pitch class of the bass note
   16. Meter: integers from 1 to 5. Lower numbers denote less accented events,
      higher numbers denote more accented events.
   17. Chord label: Chord resonating during the given event.

8. Missing Attribute Values: None

9. Class Distribution:
        D_M:   503
        G_M:   489
        C_M:   488
        F_M:   389
        A_M:   352
        BbM:   312
        E_M:   295
        A_m:   258
        E_m:   241
        B_m:   217
        G_m:   179
        D_m:   165
        EbM:   146
        C_m:   144
        F#m:   143
        B_M:   143
        F#M:    90
       C_M7:    66
       D_M7:    58
       A_M7:    56
       G_M7:    52
       B_M7:    46
       E_M7:    43
        F_m:    42
        C#M:    39
       F_M7:    38
        AbM:    37
       F#M7:    34
       D_m7:    33
        Bbm:    26
        C#m:    24
       E_m7:    24
        DbM:    21
       C_m7:    20
       B_m7:    19
       F#m7:    19
       G_m7:    18
        B_d:    17
       C_m6:    17
       D_M4:    16
       C_M4:    16
       A_M4:    16
       C#d7:    15
        F#d:    14
       E_M4:    14
       F_M4:    14
       E_m6:    14
       F#M4:    12
       D_m6:    12
        G#d:    11
       A_m7:    11
        C#d:    10
       A_m6:    10
       C#m7:     9
       G_M4:     8
       B_d7:     8
       C#M7:     7
       F_m7:     7
        D#d:     7
       F#m6:     7
        E_d:     6
        G#m:     6
       Bbm6:     6
       C_M6:     6
       G#d7:     6
        A#d:     5
        A_d:     5
        Bbd:     5
       D#d7:     4
       F_M6:     4
        Dbm:     4
       A#d7:     4
       D_d7:     4
       D_M6:     3
       F_m6:     3
       B_M4:     3
       G_M6:     3
        G_d:     3
        F_d:     3
       G_m6:     3
       Dbm7:     3
       BbM7:     3
       C#d6:     2
        Abm:     2
       C_d7:     2
       A_m4:     2
        Dbd:     2
       Dbd7:     2
       C#M4:     2
       C_d6:     2
       A_M6:     2
        D#m:     2
        D#M:     2
       B_m6:     2
        Ebd:     1
       F_d7:     1
       DbM7:     1
       EbM7:     1
        G#M:     1
       F#d7:     1
       D#d6:     1
        Abd:     1
