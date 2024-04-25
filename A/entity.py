# -*- coding: utf-8 -*-

try:
    import sys, random, copy,nltk,re
    from . import word2vecProcessing,preProcessing
    from xml.sax.saxutils import escape
    from nltk.corpus import wordnet
    from nltk.stem import SnowballStemmer
    from nltk.tag import StanfordNERTagger
    from nltk.parse.stanford import StanfordDependencyParser

except:
    sys.exit('Package Lost')
    
wn=nltk.WordNetLemmatizer()
sp=SnowballStemmer('english')


class AspectCategory:
    def __init__(self,term='',pol=''):
        self.term=term
        self.pol=pol
        
    def create(self, element):
        self.term = element.attrib['category']
        if 'polarity' in element.attrib:
            self.pol = element.attrib['polarity']
        return self

    def update(self, term='', pol=''):
        self.term = term
        self.pol = pol
        

class AspectTerm:
    def __init__(self,term='',pol='',offset=''):
        self.term=term
        self.pol=pol
        self.offset=offset
        
    def create(self, element):
        self.term = element.attrib['term']
        if 'polarity' in element.attrib:
            self.pol = element.attrib['polarity']
        self.offset = {'from': str(element.attrib['from']), 'to': str(element.attrib['to'])}
        return self

    def update(self, term='', pol='',offset=''):
        self.term = term
        self.pol = pol   
        self.offset=offset
        
# An Instance stands for a sentence
class Instance:
    def __init__(self,element):
        self.text = element.find('text').text
        self.id = element.get('id')
        self.aspect_terms = [AspectTerm('', '', offset={'from': '', 'to': ''}).create(e) for es in
                             element.findall('aspectTerms') for e in es if
                             es is not None]
        self.aspect_categories = [AspectCategory(term='', pol='').create(e) for es in element.findall('aspectCategories')
                                  for e in es if
                                  es is not None]
                                  
    def getAspectTerms(self):
        return [a.term for a in self.aspect_terms]

    def getAspectCategory(self):
        return [a.term for a in self.aspect_categories]

    def addAspectTerm(self, term, pol='', offset={'from': '', 'to': ''}):
        a = AspectTerm(term, pol, offset)
        self.aspect_terms.append(a)

    def addAspectCategory(self, term, pol=''):
        c = AspectCategory(term, pol)
        self.aspect_categories.append(c)

def fd(counts):
    d = {}
    for i in counts: 
        d[i] = d[i] + 1 if i in d else 1
    return d


# Integrate the text to a corpus
class Corpus:
    def __init__(self, elements):
        # Create a function to sort the list based on frequency
        self.freq_rank = lambda d: sorted(d, key=d.get, reverse=True)
        # To replace any special characters
        self.fix = lambda text: escape(text).replace('\"', '&quot;')
        self.corpus = [Instance(e) for e in elements]
        self.size = len(self.corpus)
        self.aspect_terms_fd = fd([a for i in self.corpus for a in i.getAspectTerms()])
        self.top_aspect_terms = self.freq_rank(self.aspect_terms_fd)
        self.texts = [t.text for t in self.corpus]

    def echo(self):
        print('%d instances\n%d distinct aspect terms' % (len(self.corpus), len(self.top_aspect_terms)))
        print('Top aspect terms: %s' % (', '.join(self.top_aspect_terms[:10])))

    def clean_tags(self):
        for i in range(len(self.corpus)):
            self.corpus[i].aspect_terms = []

    def split(self, threshold=0.8, shuffle=False):
        '''Split to train/test, based on a threshold. Turn on shuffling for randomizing the elements beforehand.'''
        clone = copy.deepcopy(self.corpus)
        if shuffle: 
            random.shuffle(clone)
        train = clone[:int(threshold * self.size)]
        test = clone[int(threshold * self.size):]
        return train, test

    # To save the corpus to a file
    def write_out(self, filename, instances, short=True):
        with open(filename, 'w') as o:
            o.write('<sentences>\n')
            #遍历每个实例
            for i in instances:
                o.write('\t<sentence id="%s">\n' % (i.id))
                o.write('\t\t<text>%s</text>\n' % self.fix(i.text))
                o.write('\t\t<aspectTerms>\n')
                if not short:
                    for a in i.aspect_terms:
                        o.write('\t\t\t<aspectTerm term="%s" polarity="%s" from="%s" to="%s"/>\n' % (
                            self.fix(a.term), a.pol, a.offset['from'], a.offset['to']))
                o.write('\t\t</aspectTerms>\n')
                o.write('\t\t<aspectCategories>\n')
                if not short:
                    for c in i.aspect_categories:
                        o.write('\t\t\t<aspectCategory category="%s" polarity="%s"/>\n' % (self.fix(c.term), c.pol))
                o.write('\t\t</aspectCategories>\n')
                o.write('\t</sentence>\n')
            o.write('</sentences>')
    
def pos_process(text):
    words=text.split(' ')
    poss=[w[1] for w in nltk.pos_tag(words)]
    return poss
    
def lemm_process(text,poss):
    words=text.split(' ')
    lemms=[]
    for i in range(len(words)):
        if poss[i][0]=='V':
            lemm=wn.lemmatize(words[i],pos='v')
        else:
            lemm=wn.lemmatize(words[i])
        lemm=sp.stem(lemm)
        lemms.append(lemm)
    return lemms
    
def w2v_cluster_processing(text,dict_w2v,max_c):
    words=text.split(' ')
    t_cluster=[]
    for word in words:
        try:
            t_cluster.append(dict_w2v[word.lower()])
        except:
            t_cluster.append(max_c+1)
    return t_cluster
  
# process dependency relation
def dep_processing(text,deps):
    # Adjective relation
    amod_l=set([])
    # Nouns relation
    nsubj_r=set([])
    #Direct object
    dobj_r=set([])

    for dep in deps:
        for triple in dep: 
            if triple[1]=='amod':
                if triple[0][0] not in amod_l:
                    amod_l.add(triple[0][0])
                    
            elif triple[1]=='dobj':
                if triple[2][0] not in dobj_r:
                    dobj_r.add(triple[2][0])
                
            elif triple[1]=='nsubj':
                if triple[2][0] not in nsubj_r:
                    nsubj_r.add(triple[2][0])
                    
    t_amod_l=[]
    t_nsubj_r=[]
    t_dobj_r=[]

    for word in text.split(' '):
        if word in amod_l:
            t_amod_l.append(True)
        else:  
            t_amod_l.append(False)
        if word in dobj_r:
            t_dobj_r.append(True)
        else:   
            t_dobj_r.append(False)
        if word in nsubj_r:
            t_nsubj_r.append(True)
        else:
            t_nsubj_r.append(False)

    return t_amod_l,t_nsubj_r,t_dobj_r
    
def isRightPosition(words,t_words,pos):
    flag=True
    i=pos
    for t_word in t_words:
        if words[i]!=t_word or i==len(words) :
            flag=False
            break
        i+=1
    return flag
            

class BIO_Entity():
    def __init__(self, instances,d_type):
        self.d_type=d_type
        self.instances = instances
        self.size = len(self.instances)
        self.origin_texts=[t.text for t in self.instances]
        self.texts = [' '.join(nltk.word_tokenize(t.text)) for t in self.instances]

        self.bio_tags=[]
        self.pos_tags=[]
        self.word_pos=[]
        self.lemm_tags=[]

        
        self.amod_l=[]
        self.nsubj_r=[]
        self.dobj_r=[]
        
        self.w2v_cluster=[]
        self.w2v_cluster_c=[]

    def createBIOTags(self):
        print('Tagging BIO')
        bios=[]

        for instance in self.instances:
            terms=instance.getAspectTerms()
            text=instance.text
            
            words=nltk.word_tokenize(text) 
            words=[re.sub(r"[^A-Za-z0-9]", "", w) for w in words]
            bio=['O' for word in words]
            for term in terms:
                t_words=nltk.word_tokenize(term)
                t_words=[re.sub(r"[^A-Za-z0-9]", "", w) for w in t_words]
                try:  
                    cur=words.index(t_words[0])
                    if isRightPosition(words,t_words,cur)==False:
                        cur=words.index(t_words[0],cur+1)
                    bio[cur]='B'
                    for i in range(1,len(t_words)):
                        bio[cur+i]='I'
                except:
                    print("\r\nFail to find AspectTerm in\n\nText: %s\n\nAspectTerm: %s"%(text, term))
            bios.append(bio)
        self.bio_tags=bios
        
    def createPOSTags(self):
        print('Marking part of speech')
        t_pos_tags=[pos_process(text) for text in self.texts]
        self.pos_tags=t_pos_tags   
        
    def createLemm(self):
        print('Tagging root of words')
        all_lemms=[]
        for i in range(len(self.texts)):
            all_lemms.append(lemm_process(self.texts[i],self.pos_tags[i]))
        self.lemm_tags=all_lemms
        
    def createW2VCluster(self):
        print('Tagging W2V cluster')
        dict_W2V=word2vecProcessing.loadDict('B/cluster/%s_w2v.pkl'%self.d_type)
        max_c=max([item[1] for item in dict_W2V.items()])
        cluster=[w2v_cluster_processing(text,dict_W2V,max_c) for text in self.texts]
        self.w2v_cluster=cluster
        
        print('Tagging W2V CBOW')
        dict_W2V=word2vecProcessing.loadDict('B/cluster/%s_w2v_c.pkl'%self.d_type)
        max_c=max([item[1] for item in dict_W2V.items()])
        cluster=[w2v_cluster_processing(text,dict_W2V,max_c) for text in self.texts]
        self.w2v_cluster_c=cluster
        
    def createDependenceFeature(self,dep_path):
        print('Tagging dependence information')
        depend_list=preProcessing.loadDependenceInformation(dep_path)

        dep_amod_l=[]
        dep_nsubj_r=[]
        dep_dobj_r=[]
        
        for i in range(len(self.texts)):
            t_amod_l,t_nsubj_r,t_dobj_r=dep_processing(self.texts[i],depend_list[i])
            
            dep_amod_l.append(t_amod_l)
            dep_nsubj_r.append(t_nsubj_r)
            dep_dobj_r.append(t_dobj_r)

        self.amod_l=dep_amod_l
        self.nsubj_r=dep_nsubj_r
        self.dobj_r=dep_dobj_r
        
    def getFeaturesAndLabels(self):
        print('Getting features')
        features=[]
        for i in range(self.size):
            feature=[]
            text=self.texts[i]
            text=text.split(' ')
            for j in range(len(text)):
                feature.append({'word':text[j],'pos':self.pos_tags[i][j],
                                'lemm':self.lemm_tags[i][j],
                                'w2v_c':self.w2v_cluster[i][j],
                                'w2v_c_c':self.w2v_cluster_c[i][j],

                                'amod_l':self.amod_l[i][j],
                                'nsubj_r':self.nsubj_r[i][j],
                                'dobj_r':self.dobj_r[i][j],
                                })
            features.append(feature)
            
            if not (len(text)==len(self.pos_tags[i]) and len(self.pos_tags[i])==len(self.bio_tags[i])):
                print('Fail to match a feature')
        return features,self.bio_tags
                
    def clean_tags(self):
        for i in range(len(self.corpus)):
            self.corpus[i].aspect_terms = []
        