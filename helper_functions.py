
import pandas as pd 
from sklearn.mixture import GaussianMixture 
import matplotlib.pyplot as plt 

def scaled(df):
    from sklearn.preprocessing import StandardScaler, normalize
    import pandas as pd 
    x_scaled = normalize(df)
    df_scaled=pd.DataFrame(x_scaled,columns=list(df.columns))
    return df_scaled 

def pca(data,n):
    from sklearn.decomposition import PCA
    
    """This function is used to reduced the dimention of data frame """
    
    pca = PCA(n)
 
    df_pca = pca.fit_transform(data)
    
    principalDf = pd.DataFrame(data =df_pca
             , columns = ['principal component 1', 'principal component 2'])
    
    return principalDf

def scatter(df,x,y,label):
    """Scatter plot betwween x and y variables """
    import seaborn as sns
    sns.scatterplot(data=df,x=x,y=y,hue=label)
    
def bic(X):

  bic_scores = []
  no_of_comp=[]
  for i in range (1,10):
      #print(i)
      gm = GaussianMixture(n_components=i, random_state=0).fit(X)
      #print(gm.bic(X_train))
      bic_scores.append(gm.bic(X))
      no_of_comp.append(i)
      plt.plot(no_of_comp,bic_scores) 
  plt.show()
#   print(bic_scores)
#   print(no_of_comp)


def missing_details(df):
    b = pd.DataFrame()
    b['Missing value, %'] = round(df.isnull().sum()/df.shape[0]*100)
    b['Missing value count']=df.isnull().sum()
    b['N unique value'] = df.nunique()
    return b 

    
    
    
    
    


