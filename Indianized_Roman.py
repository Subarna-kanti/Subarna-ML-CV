dict = {
    'त्र':'tr', 
'ज्ञ':'jñ', 
'श्र':'śr',
'क़':'q',
'ख़':'k͟h',
'ग़':'ġ',
'क़्':'q',
'ख़्':'k͟h',
'ग़्':'ġ',
'फ़्':'f',
'ज़':'z',
'ज़्':'z',
'फ़':'f',
'ड़':'ṛ',
'ढ़':'ṛh',
'अ':'a',
'आ':'ā',
'ा':'ā',
'इ':'i',
'ि':'i',
'ई':'ī' ,
'ी':'ī' ,
'उ':'u' ,
'ऊ':'ū' ,
'ए':'e' ,
'ऐ':'ai' ,
'ओ':'o' ,
'औ':'au',
'ु':'u' ,
'ू':'ū' ,
'े':'e' ,
'ै':'ai' ,
'ो':'o' ,
'ौ':'au',
'ऋ ':'ṛ',
'ॠ ':'ṝ',
'ऌ ':'ḷ',
'ॡ':'ḹ',
'अं ':'ṃ' ,
'अः' :'ḥ',
'ं ':'m̐' ,
'ः' :'ḥ',
'अँ ':'m̐' ,
'ँ':'m̐' ,
'क':'k' ,
'ख':'kh',
'ग':'g',
'घ':'gh',
'ङ':'ṅ' ,
'च':'c' ,
'छ':'ch',
'ज':'j' ,
'झ':'jh',
'ञ':'ñ' ,
'ट':'ṭ' ,
'ठ':'ṭh',
'ड':'ḍ' ,
'ढ':'ḍh',
'ण':'ṇ' ,
'त':'t' ,
'थ':'th',
'द':'d' ,
'ध':'dh',
'न':'n',
'प':'p',
'फ':'ph',
'ब':'b' ,
'भ':'bh',
'म':'m' ,
'य':'y' ,
'र':'r' ,
'ल':'l' ,
'व':'v',
'श':'ś',
'स':'s' ,
'ह':'h',
'क्ष':'kṣ',
       }

while(5):
    string = input("enter word: ")
    if(string == "exit"):
        break
    output_s = ""
    i = 0
    while(5):
        if(i >= len(string)):
            break
        split1 = [string[i:i+2] for i in range(i, len(string), 2)]
        split2 = [string[i:i+1] for i in range(i, len(string), 1)]
        if(split2[0] == '्'):
            i = i + 1
            continue
        for key,values in dict.items():
            if(split1[0] == key):
                output_s = output_s + values
                i = i + 2
                break
            elif(split2[0] == key):
                output_s = output_s + values
                i = i + 1
                break
    print(output_s)

