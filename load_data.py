import pathlib
import glob
import pickle

if __name__ == "__main__":
    # Load All Data: download from: http://opus.nlpl.eu/OpenSubtitles.php
    # http://opus.lingfil.uu.se/download.php?f=OpenSubtitles/en.tar.gz
    # Please cite the following article if you use any part of the corpus in your own work:
    # Jörg Tiedemann, 2009, News from OPUS - A Collection of Multilingual Parallel Corpora with Tools and Interfaces. In N. Nicolov and K. Bontcheva and G. Angelova and R. Mitkov (eds.) Recent Advances in Natural Language Processing (vol V), pages 237-248, John Benjamins, Amsterdam/Philadelphia


    all_dialogues = list(pathlib.Path('OpenSubtitles/').glob('**/*.xml.gz'))
    dialogue_lines = []
    counter = 0
    
    for dialogue in all_dialogues:
        print(counter)
        content = gzip.open( str(dialogue)).read()
        # content = infile.read()
        root = parseString(content).documentElement
        itemlist = root.getElementsByTagName('s')
        for item in itemlist:
            sent = []
            for word in item.getElementsByTagName('w'):
                sent.append(word.firstChild.data)
            
            dialogue_lines.append(' '.join(sent))
        counter+=1
        print(str(infile)+"-----done.")

    pickle.dump( dialogue_lines, open( "dialogues.p", "wb" ) )