import pandas as pd
import re

class preprocessing_dataframe(pd.DataFrame):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # Remove label which only has few samples
    def remove_label(self, labels=None):
        for label in labels:
            for i in self['Intent']:
                if label in i:
                    tag = self['Intent'] == i
                    self.drop(self[tag].index, inplace = True)

        self.reset_index(drop=True, inplace=True)

    # Remove sentence which has length greater than 40
    def remove_higher_len(self, len_threshold=40):
        for i in self['Question']:
            if len(i.split()) > len_threshold:
                filter = self['Question'] == i
                self.drop(self[filter].index, inplace=True)

        self.reset_index(drop=True, inplace=True)

    # Replacing index, label and replace them with question
    def preprocessing_ner(self):
        list_process_ques = []
        list_labels = []

        for i in range(self.shape[0]):
            para = self['Parameters'][i]
            lists = re.findall(r'\[[^]]*', para)
            list_index = [x.replace('[','').split(',') for x in lists]
            try:
                list_index = sorted(list_index, key = lambda x: int(x[0]))
            except:
                pass

            sentence = self['Question'][i]
            new_sentence = ""
            current_index = 0
            for i in list_index:
                try:
                    start = int(i[0])
                    end = int(i[1])
                    label = str(i[2].replace('"', '').strip())
                    list_labels.append(label)
                    new_sentence += sentence[current_index:start] + label
                    current_index = end
                except:
                    pass
            new_sentence += sentence[current_index:]
            list_process_ques.append(new_sentence)

        self['Preprocess_Question'] = list_process_ques
        self['Question'] = self['Question'].apply(lambda x: x.strip())

        return set(list_labels)
    
    # For target column, if not in list it would return 0
    def create_target(self, list_labels):
        list_Y = []
        for sent in self['Preprocess_Question']:
            pre_sent = [x if x in list_labels else '0' for x in sent.strip().split(' ')]
            combine = " ".join(pre_sent)
            list_Y.append(combine)

        self['Target'] = list_Y

    # Splitting sentence into a list contains a word, character and label
    def split_word_char_label(self):
        sentence = []
        for idx, sent in enumerate(self['Question']):
            temp = []
            for word in sent.split(" "):
                temp.append([word])
            sentence.append(temp)
        
        temp_set = sentence.copy()

        for i, sentence in enumerate(sentence):
            label = self['Target'][i].split(" ")
            for j, word in enumerate(sentence):
                temp_set[i][j].append([c for c in word[0]])
                temp_set[i][j].append(label[j])
            
        self['input'] = temp_set
        