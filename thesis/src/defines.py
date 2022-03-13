## thesis defines
START_CHAR = '&'
END_CHAR = '#'
LABEL = 'is_nar'
THERAPIST_HEB = "מטפל"
CLIENT_HEB = "קליינט"
SEGMENT_HEB = "סגמנט"
PATH_TO_DFS = "dataframes"
CRF_SEQ_LEN=6
CRF_SEQ_STEP=6
CRF_NEIGHBOR_RADIUS=2
YAP_TAG_DICT = {
'POS_TAGS_DICT' : {
'AGR-gn' : 'Agreement particle',
'AT' : 'Accusative marker',
'AUX' : 'Auxiliary verb',
'CC' : 'Coordinating conjunction',
'CD-gn-(H)' : 'Numeral (definite)',
'CDT-gn-(H)' : 'Numeral determiner (definite)',
'COM' :  'Complementizer',
'DT' : 'Determiner',
'IN' : 'Preposition',
'JJ-gn-(H)' : 'Adjective (definite)',
'JJT-gn' : 'Construct state adjective',
'H' : 'Definiteness marker',
'HAM' : 'Yes/No question word',
'MD-gnpt' : 'Modal',
'MOD' : 'Modifier',
'NN-gn-(HjH-gnp)' : 'Noun (definitejdefinite-genitive)',
'NNG-gn-(HjH-gnp)' : 'Gerund noun (definitejdefinite-geni',
'NNGT-gn' : 'Construct state gerund',
'NNP-gn' : 'Proper noun',
'NNT-gn' : 'Construct state noun',
'POS' : 'Possessive item',
'PRP-gnp' : 'Personal pronoun',
'QW' : 'Question/WH word',
'RB' : 'Adverb',
'RBR' : 'Adverb, comparative',
'REL' : 'Relativizer',
'VB-gnpt' : 'Verb, finite',
'VB-M' : 'Verb, infinite',
'WDT-gn' : 'Determiner question word',
'ZVL' : 'Garbage',
'yy' :  'various symbols, see appendix A'
},
'FUNC_FEATURES_DICT' : {
'SBJ' : 'subject',
'OBJ' : 'object',
'COM' : 'complement ',
'ADV' : 'adverbial',
'CNJ' : 'conjunction'
},
'AGREEMENT_FEATURES_DICT' : {
'gen' : 'gender', # [M,F]
'num' : 'number', # [S=singular,P=plural]
'p': 'person', #[1,2,3,A=1,2,3]
'tense' : 'tense' #V=past, H=present, T=future, C=imperative
},
'SYNT_TAG_DICT' : {
'ADJP-gn-(H)' : 'Adjective phrase',
'ADVP' : 'Adverb phrase',
'FRAG' : 'Fragment of a declarative sentence',
'FRAGQ' : 'Fragment of an interrogative sentence',
'INTJ' : 'Interjection',
'NP-gn-(H)' : 'Noun phrase',
'PP' : 'Preposition phrase',
'PREDP' : 'Predicate phrase',
'PRN' : 'Parenthetical',
'S' : 'Declarative sentence',
'SBAR' : 'Clause introduced by a COM, REL or IN word',
'SQ' : 'Interrogative sentence',
'VP' : 'Verb phrase',
'VP-MD' : 'Verb phrase with a modal verb',
'VP-INF' : 'Verb phrase with an infinitival verb'
},
'MY_POS_TAG_DICT' :
{
'NNT' : 'Noun in construct state form',
'INTJ' : 'Interjection',
'NNP' : 'Proper noun',
'QW' : 'WH words like when, where and how, which do not appear in a determiner position',
'DEF' : 'Determiner',
'NN' : 'NOUN',
'PREPOSITION' : 'PREPOSITION',
'REL' : 'The relativizers she, aher and ha (=that)',
'JJ' : 'The construct state form of adjectives (like pitiless)',
'RB' : 'Adverb',
'COP' : 'Auxiliary verb = cop',
'CC' : 'Coordinating conjunction',
'PRP' : 'Personal pronoun',
'VB' : 'A verb',
'IN' : 'Preposition',
'DTT' : 'definite article', # TBD confirm
'TEMP' : 'TEMP', # TBD find out what it is
'CONJ' : 'conjunction',
'BN' : 'VERB-VerbForm=Part',
'EX' : ' VERB-HebExistential=True',
'DT' : 'definite article',   # TBD confirm
'POS' : 'Possessive item',
'S_PRN' : 'Declarative sentence Parenthetical',
'AT' : 'Accusative marker',
'CD' : 'Numeral',
'MD' : 'Modal'
}
}
HEB2UDPOS_DICT = {
		"AT":       "PART-Case=Acc",
		"BN":       "VERB-VerbForm=Part",
		"BNT":      "VERB-Definite=Cons|VerbForm=Part",
		"CC":       "CCONJ",
		"CC-SUB":   "SCONJ",
		"CC-COORD": "SCONJ",
		"CC-REL":   "SCONJ",
		"CD":       "NUM",
		"CDT":      "NUM-Definite=Cons",
		"COP":      "AUX-VerbType=Cop",
		"DT":       "DET-Definite=Cons",
		"DTT":      "DET-Definite=Cons",
		"EX":       "VERB-HebExistential=True",
		"IN":       "ADP",
		"INTJ":     "INTJ",
		"JJ":       "ADJ",
		"JJT":      "ADJ-Definite=Cons",
		"MD":       "AUX-VerbType=Mod",
		"NEG":      "ADV",
		"NN":       "NOUN",
		"NNP":      "PROPN",
		"NNT":      "NOUN-Definite=Cons",
		"NNPT":     "PROPN-Abbvr=Yes",
		"P":        "ADV-Prefix=Yes",
		"POS":      "PART-Case=Gen",
		"PRP":      "PRON",
		"QW":       "ADV-PronType=Int",
		"RB":       "ADV-Polarity=Neg",
		"TTL":      "NOUN-Title=Yes",
		"VB":       "VERB"
}
