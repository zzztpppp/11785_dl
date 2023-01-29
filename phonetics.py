CMUdict_ARPAbet = {
    "[SIL]": "-", "NG": "G", "F" : "f", "M" : "m", "AE": "@", 
    "R"    : "r", "UW": "u", "N" : "n", "IY": "i", "AW": "W", 
    "V"    : "v", "UH": "U", "OW": "o", "AA": "a", "ER": "R", 
    "HH"   : "h", "Z" : "k", "K" : "k", "CH": "C", "W" : "w", 
    "EY"   : "e", "ZH": "t", "T" : "t", "EH": "E", "Y" : "y", 
    "AH"   : "A", "B" : "b", "P" : "p", "TH": "T", "DH": "D", 
    "AO"   : "c", "G" : "g", "L" : "l", "JH": "j", "OY": "O", 
    "SH"   : "S", "D" : "d", "AY": "Y", "S" : "s", "IH": "I",
    "[PAD]": "[PAD]" , "[SOS]": "[SOS]", "[EOS]": "[EOS]"}

CMUdict = list(CMUdict_ARPAbet.keys())
ARPAbet = list(CMUdict_ARPAbet.values())

Alphabet = [ 
    '[SIL]',
    'a',
    'b',
    'c',
    'd',
    'e',
    'f',
    'g',
    'h',
    'i',
    'j',
    'k',
    'l',
    'm',
    'n',
    'o',
    'p',
    'q',
    'r',
    's',
    't',
    'u',
    'v',
    'w',
    'x',
    'y',
    'z',
    ' ',
    "'",
    '[PAD]',
    '[SOS]',
    '[EOS]']

VOCAB = ['<sos>',
         'A',   'B',    'C',    'D',
         'E',   'F',    'G',    'H',
         'I',   'J',    'K',    'L',
         'M',   'N',    'O',    'P',
         'Q',   'R',    'S',    'T',
         'U',   'V',    'W',    'X',
         'Y',   'Z',    "'",    ' ',
         '<eos>']

SOS_TOKEN = 0
EOS_TOKEN = len(VOCAB) - 1
