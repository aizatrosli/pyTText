import pandas as pd


# function to remove special characters , punctions ,stop words ,
# digits ,hyperlinks and case conversion
def string_manipulation(df, column):
    # extract hashtags
    df["hashtag"] = df[column].str.findall(r'#.*?(?=\s|$)')
    # extract twitter account references
    df["accounts"] = df[column].str.findall(r'@.*?(?=\s|$)')

    # remove hashtags and accounts from tweets
    df[column] = df[column].str.replace(r'@.*?(?=\s|$)', " ")
    df[column] = df[column].str.replace(r'#.*?(?=\s|$)', " ")

    # convert to lower case
    df[column] = df[column].str.lower()
    # remove hyperlinks
    df[column] = df[column].apply(lambda x: re.split('https:\/\/.*', str(x))[0])
    # remove punctuations
    df[column] = df[column].str.replace('[^\w\s]', " ")
    # remove special characters
    df[column] = df[column].str.replace("\W", " ")
    # remove digits
    df[column] = df[column].str.replace("\d+", " ")
    # remove under scores
    df[column] = df[column].str.replace("_", " ")
    # remove stopwords
    df[column] = df[column].apply(lambda x: " ".join([i for i in x.split()
                                                      if i not in (stop_words)]))
    return df

positive_words = pd.read_csv(r"../input/positive-words/positive-words.txt",
                             header=None)
#negative words
negative_words = pd.read_csv(r"../input/negative-words/negative-words.txt",
                             header=None,encoding='latin-1')

#convert words to lists
def convert_words_list(df) :
    words = string_manipulation(df,0)
    words_list = words[words[0] != ""][0].tolist()
    return words_list

positive_words_list = convert_words_list(positive_words)

#remove word trump from positive word list
positive_words_list = [i for i in positive_words_list if i not in "trump"]
negative_words_list = convert_words_list(negative_words)

print ( "positive words : " )
print (positive_words_list[:50])
print ( "negative words : " )
print (negative_words_list[:50])