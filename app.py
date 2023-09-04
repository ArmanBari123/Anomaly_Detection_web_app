import pandas as pd 
import numpy as np 
import scipy.io as sio
import streamlit as st
import helper_functions as hp 
import plotly.figure_factory as ff
import plotly.express as px
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.mixture import GaussianMixture
from sklearn import metrics
from scipy.spatial.distance import mahalanobis
import altair as alt


import mat73
import h5py
from PIL import Image
st.set_page_config(layout="wide")




#initializing the containers 
header=st.container()
data_set=st.container()
features=st.container()
Model_training=st.container()


@st.cache

def get_data_mat1(filename):

    test = sio.loadmat(filename)
    df = pd.DataFrame(np.hstack((test['X'], test['y'])))
    return df 
    
def get_data_mat2(filename):

    data_dict = mat73.loadmat(filename)
    x=pd.DataFrame(data_dict["X"])
    y=pd.DataFrame(data_dict["y"])
    df1=pd.concat([x,y],axis=1)
    df1.columns=['x1','x2','x3','y']
    return df1


with header:
    st.title("Anomaly Detection WebApp") 
    ##st.text("describe the project")

    
    image = Image.open('image.png')
    st.image(image, caption='finding the anomal fish',width=None,use_column_width=200)






with data_set:
   
    data=st.selectbox(
    'Select the data set',
    ('annthyroid.mat', 'http.mat', 'cover.mat','thyroid.mat','satellite.mat','upload_data'))
    
    if data=='cover.mat':

        df=get_data_mat1(data)
        st.header("Forest data set ")
        st.subheader("Dataset Information")
        st.markdown('Domain :**Taxonomy** \n ')
        
        st.markdown("The original ForestCover/Covertype dataset from UCI machine learning repository is a multiclass classification dataset. It is used in predicting forest cover type from cartographic variables only (no remotely sensed data). This study area includes four wilderness areas located in the Roosevelt National Forest of northern Colorado. These areas represent forests with minimal human-caused disturbances, so that existing forest cover types are more a result of ecological processes rather than forest management practices. This dataset has 54 attributes (10 quantitative variables, 4 binary wilderness areas and 40 binary soil type variables). Here, outlier detection dataset is created using only 10 quantitative attributes. Instances from class 2 are considered as normal points and instances from class 4 are anomalies. The anomalies ratio is 0.9%. Instances from the other classes are omitted.")
        st.write("http://odds.cs.stonybrook.edu/forestcovercovertype-dataset/")

    elif data=='http.mat':

        df=get_data_mat2(data)
        st.header("Network data set ")
        st.subheader("Dataset Information")
        st.markdown('Domain :**INTRUSION DETECTOR LEARNING** \n ')
        
        st.markdown("The original KDD Cup 1999 dataset from  UCI machine learning repository contains 41 attributes (34 continuous, and 7 categorical), however, they are reduced to 4 attributes (service, duration, src_bytes, dst_bytes) as these attributes are regarded as the most basic attributes (see kddcup.names), where only ‘service’ is categorical. Using the ‘service’ attribute, the data is divided into {http, smtp, ftp, ftp_data, others} subsets. Here, only ‘http’ service data is used. Since the continuous attribute values are concentrated around ‘0’, we transformed each value into a value far from ‘0’, by y = log(x + 0.1). The original data set has 3,925,651 attacks (80.1%) out of 4,898,431 records. A smaller set is forged by having only 3,377 attacks (0.35%) of 976,157 records, where attribute ‘logged_in’ is positive. From this forged dataset 567,497 ‘http’ service data is used to construct the http (KDDCUP99) dataset")
        st.write("http://odds.cs.stonybrook.edu/http-kddcup99-dataset/")
       


    elif data=='annthyroid.mat':
        df=get_data_mat1(data)
        st.header("Health Care data set ")
        st.subheader("Dataset Information")
        st.markdown('Domain :**Health Care anomaly** \n ')
        st.markdown("Thyroid Disease Data Set Dataset information The original arrhythmia dataset from UCI machine learning repository is a multi-class classification dataset with dimensionality 279. There are five categorical attributes which are discarded here, totalling 274 attributes. The smallest classes, i.e., 3, 4, 5, 7, 8, 9, 14, 15 are combined to form the outliers class and the rest of the classes are combined to form the inliers class.")

    elif data =='satellite.mat':
        df=get_data_mat1(data)
        st.header("Setellite data set ")
        st.subheader("Dataset Information")
        st.markdown('Domain :**Landsat Satellite** \n ')
        st.markdown("The database consists of the multi-spectral values of pixels in 3x3 neighbourhoods in a satellite image, and the classification associated with the central pixel in each neighbourhood. The aim is to predict this classification, given the multi-spectral values. In the sample database, the class of a pixel is coded as a number.The original Statlog (Landsat Satellite) dataset from UCI machine learning repository is a multi-class classification dataset. Here, the training and test data are combined. The smallest three classes, i.e. 2, 4, 5 are combined to form the outliers class, while all the other classes are combined to form an inlier class. ")
        st.write("http://odds.cs.stonybrook.edu/satellite-dataset/")

    elif data=='thyroid.mat':

        df=get_data_mat1(data)
        st.header("Health Care data set ")
        st.subheader("Dataset Information")
        st.markdown('Domain :**Health Care anomaly** \n ')
        st.markdown("The original thyroid disease (ann-thyroid) dataset from UCI machine learning repository is a classification dataset,The problem is to determine whether a patient referred to the clinic is hypothyroid.")

    #elif data=='upload_data':

        # from io import StringIO
        # uploaded_file = st.file_uploader("Choose a file")
        # if uploaded_file is not None:


        #     # To read file as bytes:
        #     bytes_data = uploaded_file.getvalue()
        #     st.write(bytes_data)

        #     # To convert to a string based IO:
        #     stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
        #     st.write(stringio)

        #     # To read file as string:
        #     string_data = stringio.read()
        #     st.write(string_data)

        #     # Can be used wherever a "file-like" object is accepted:
        #     df= pd.read_csv(uploaded_file)
        #     st.write(df)








    checkbox = st.sidebar.checkbox("View Raw data.")
    if checkbox:
        st.subheader("Orignal Data")
        st.dataframe(data=df)
        st.subheader("Data Shape")
        rows=df.shape[0]
        col=df.shape[1]
        st.metric(label="No of rows", value= rows)
        st.metric(label="No of Columns", value= col)


    checkbox_miss=st.sidebar.checkbox("Missing Value Analysis")
    if checkbox_miss:
        st.subheader("Missing value")
        st.dataframe(data=hp.missing_details(df))
    
    checkbox_desc=st.sidebar.checkbox("Data Summary stats")
    if checkbox_desc:
        st.subheader("Summary statistics")
        st.dataframe(data=df.describe())


    #     st.subheader("Data Shape")
    #     rows=df.shape[0]
    #     col=df.shape[1]
    #     st.metric(label="No of rows", value= rows)
    #     st.metric(label="No of Columns", value= col)
        

    #Dimentionality reduction 
    x=df.iloc[:,:-1]
    y=df.iloc[:,-1]
    dfscaled=hp.scaled(x)
    dfpca=hp.pca(dfscaled,2)
    dfpca['Labels']=y

    #Dimentionality reduction
    checkbox2 = st.sidebar.checkbox("View PCA Output")
    if checkbox2:
        # st.write(data)
        st.subheader("Dataset at Low Dimention Space")
        st.dataframe(data=dfpca)

    #visualize data 
    checkbox3 = st.sidebar.checkbox("Visualize data at Low Dimention Space")
    if checkbox3:
        st.subheader("Scatter Plot ")
        numeric_columns=list(dfpca.columns)
        #select_box1 = st.sidebar.selectbox(label='X axis', options=numeric_columns)
        #select_box2 = st.sidebar.selectbox(label="Y axis", options=numeric_columns)
        fig= sns.relplot(x='principal component 1', y='principal component 2', data=dfpca,hue=y)
        st.pyplot(fig)

    # % outliers 
    checkbox4=st.sidebar.checkbox("Check the class imbalance")
    if checkbox4:
        st.subheader("% Outliers")
        numeric_columns=list(dfpca.columns)
        select_label=st.sidebar.selectbox(label='select label column',options=numeric_columns)
        outliers_labels=list(dfpca[select_label].unique())
        select_label_val=st.sidebar.selectbox(label='select label value',options=outliers_labels)
        percent=(len(dfpca[dfpca[select_label]==select_label_val])/len(dfpca))*100
        st.metric(label="% Ouliers", value= round(percent,2))

        # we will add more feature EDA PART after Model training 

    if st.button('Click Here for Training Guidline'):
        st.markdown("Users of this framework must ensure that the objective of this framework is to maximize the likelihood of detecting outliers based on the joint probability distribution of the data Therefore selection of hyperparameters is the real crux of this framework")
        st.markdown("**How do we select the hyperparameters?**")
        st.markdown("- Use the elbow curve of Baysian Inference Criteria (BIC) and select the number of components where the elbow of the curve is visible.")
        st.markdown("- We can further increase the number of components until we do not get the maximum likelihood cluster of the anomalous data.")
        st.markdown("- Change the covariance type and monitor % outliers (test), and select the covariance type where we get high % outliers for any component. but make sure the corresponding % of inliers should be minimum.")
     
        
with Model_training:

    st.header("Time to Train The Model!")
    st.text("Here you will chose the hyperparameters of GMM and see how the performance changes")
        
        
        
        # Seperating Ouliers and Inliers
    dfscaled['Label']=y
    inliers=dfscaled[dfscaled['Label']==0] 
    outliers=dfscaled[dfscaled['Label']==1]
    X_outlier=outliers.iloc[:,:-1]
    X_inlier=inliers.iloc[:,:-1]

    Split_ratio = st.slider('Chose the train test split ratio?', 0.1, 0.9, 0.1)

    #train test split
    X_train, X_test = train_test_split(X_inlier, test_size=Split_ratio, random_state=1)

        #elbow=hp.bic(X_train)
    st.set_option('deprecation.showPyplotGlobalUse', False)
    #st.pyplot(elbow)
    st.subheader('Guidline for optimum number of Components')

    ## if st.button('Display Elbow chart to select optimum number of components '):
    bic_scores = []
    no_of_comp=[]
    for i in range (1,10):
      gm = GaussianMixture(n_components=i, random_state=0).fit(X_train)
      bic_scores.append(gm.bic(X_train))
      no_of_comp.append(i)
    d = {'bic_scores': bic_scores, 'no_of_comp':no_of_comp }
    BIC=pd.DataFrame(d)
    
    st.write(BIC)
    st.subheader("Elbow curve ")
    st.text("BIC score vs Number of components")
    st.markdown("Pick the elbow of the curve as the number of components to use.")
    st.line_chart(BIC)


    components = st.slider('select the Number of components?', 1, 20, 1)
    cov_type = st.selectbox('Select the covariance type', ("spherical", "diag", "tied", "full"))

    if st.button('Click Here to Train your Model'):
    #fit model 
        gm = GaussianMixture(n_components=components, random_state=0,covariance_type=cov_type).fit(X_train)
        train_components=gm.predict(X_train)
        test_components=gm.predict(X_test)
        #Threshold_Hard_Member = st.slider('select the threshold value of propabity for defining hard and soft members?', 0.1, 0.99, 0.1)

        st.subheader("Train Test Performance")
        train_prob=gm.predict_proba(X_train).round(1)
        Hard_member=np.any(train_prob >0.9, axis=1)
        Hard_members=np.array(np.where(Hard_member==True)).size
        Total_Train=train_prob.shape[0]
        soft_members=Total_Train-Hard_members
        percent_hard_Members=(Hard_members/Total_Train)*100
        percent_soft_members=(soft_members/Total_Train)*100
            
        #st.metric(label="silhouette score", value= metrics.silhouette_score(X_train,train_components))
        

        test_prob=gm.predict_proba(X_test).round(1)
        Hard_member_test=np.any(test_prob >0.9, axis=1)
        Hard_members_test=np.array(np.where(Hard_member_test==True)).size
        Total_Test=test_prob.shape[0]
        soft_members_test=Total_Test-Hard_members_test
        percent_Test_hard_Members=(Hard_members_test/Total_Test)*100
        percent_Test_soft_Members=(soft_members_test/Total_Test)*100

        c1, c2, c3, c4 = st.columns(4)
        c1.metric(label="% Hard Members Train", value= round(percent_hard_Members,1))
        c2.metric(label="% Soft Members Train ", value=round(percent_soft_members,1))
        c3.metric(label="% Hard Members Test", value= round(percent_Test_hard_Members,1))
        c4.metric(label="% Soft Members Test", value=round(percent_Test_soft_Members,1))
        
        # col1, col2, col3 = st.columns(3)

        st.text("Any point whose probabilily value greater than 0.9 will be marked as Hard member")

        # c5, c6 = st.columns(2)
        # train_score=metrics.silhouette_score(X_train,train_components)
        # test_score=metrics.silhouette_score(X_test,test_components)
        # c5.metric(label="silhouette score Train", value=round(train_score,1))
        # c6.metric(label="silhouette score Train", value=round(test_score,1))

        # c5, c6 = st.columns(2)
        # train_score=metrics.calinski_harabasz_score(X_train,train_components)
        # test_score=metrics.calinski_harabasz_score(X_test,test_components)
        # # c5.metric(label="calinski_harabasz_score Train", value=round(train_score,1))
        # c6.metric(label="calinski_harabasz_score Test", value=round(test_score,1))
        
        #Predicting Ouliers
        ouliers_comp=gm.predict(X_outlier)
        Anomaly_prob=gm.predict_proba(X_outlier).round(1)
        Anomaly_Hard_member=np.any(Anomaly_prob >0.9, axis=1)
        Anomaly_Hard_members=np.array(np.where(Anomaly_Hard_member==True)).size
        Total_Anamolies =Anomaly_prob.shape[0]
        Anomaly_soft_members=Total_Anamolies-Anomaly_Hard_members
        percent_Anomaly_hard_Members=(Anomaly_Hard_members/Total_Anamolies)*100
        percent_Anomaly_soft_Members=(Anomaly_soft_members/Total_Anamolies)*100
        c5, c6 = st.columns(2)
        c5.metric(label="% Hard Members Anomalies", value= round(percent_Anomaly_hard_Members,1))
        c6.metric(label="% Soft Members Anomalies ", value=round(percent_Anomaly_soft_Members,1))

        #reducing the dimention of the data for visualization purpose
        min_train=hp.pca(X_train,2)
        min_test=hp.pca(X_test,2)
        min_outlier=hp.pca(X_outlier,2)

        
        c7,c8,c9 = st.columns([4,4,4])
        fig1= sns.relplot(x='principal component 1', y='principal component 2', data=min_train,hue=train_components)
        fig2= sns.relplot(x='principal component 1', y='principal component 2', data=min_test,hue=test_components)
        fig3= sns.relplot(x='principal component 1', y='principal component 2', data=min_outlier,hue=ouliers_comp)
        c7.subheader("Taining Clusters")
        c7.pyplot(fig1)
        c8.subheader("Test Clusters")
        c8.pyplot(fig2)
        c9.subheader("Outlier Clusters")
        c9.pyplot(fig3)
     # creating table for summarizing cluster results 

        train=pd.value_counts(train_components)/len(train_components)*100
        test=pd.value_counts(test_components)/len(test_components)*100
        outliers=pd.value_counts(ouliers_comp)/len(ouliers_comp)*100
        d = {'% Inliers (Train)': round(train,1), '% Inliers (Validation) ': round(test,1),'% Outliers Test':round(outliers,1)}

        df_results=pd.DataFrame(d)
        df_results.index.name = 'Gausian Components'
        df_results=df_results.fillna(0)
        st.subheader("Components Summary")
        st.write(df_results)
        















        #Statistical Distance 
        #adding the predicted labels to the data frame 
        #train
        df_train=X_train.copy()
        df_train["Gausian Components"]=train_components
        #test
        df_test=X_test.copy()
        df_test["Gausian Components"]=test_components
        #combining train test data set 
        df_predictions=pd.concat([df_test, df_train],ignore_index=True)

        # creating a Data frame of Mean Vector of each component
        features_list=list(df_predictions.iloc[:,:-1].columns) # removing labels for mean vector 
        Mean_Vector=df_predictions.groupby(['Gausian Components'])[features_list].mean()
        #Unique list of components 
        components_list=[]
        for i ,row in Mean_Vector.iterrows():
            components_list.append(i)
       
       
         # Creating a covariance Matrix  of each components 
        features_list=list(df_predictions.iloc[:,:-1].columns) #list of X feature except labels 
        covariance_matrix=df_predictions.groupby(['Gausian Components'])[features_list].cov() # passing list of x features to get the covariance matrix of each components
        

        #Creating a inverse covariance Martix array of each components 
        inv_cov=[] 
        for i in components_list:  #iterate over each components 
            inv_cov.append(pd.DataFrame(np.linalg.inv(covariance_matrix.loc[i])))
        #st.write(inv_cov)
        def Avg_Distance(component):
            from scipy.spatial.distance import mahalanobis
            for index, row in X_outlier.iterrows():

                inputs=row.values
                dist=mahalanobis(inputs, Mean_Vector.loc[component], inv_cov[component])
                dist=dist.mean()
                return dist

        Average_distance=[]
        for i in components_list:
            Average_distance.append(Avg_Distance(i))

        
        st.subheader("Statistical Distance")
        st.bar_chart(Average_distance)
        st.caption("Average Mahalonabis distance from Components")



        components_list=['Com1','comp2','comp3','comp4']
        data = {'Gausian Comonents': components_list,
        'Avg_Mahalanobis_Distance': Average_distance}
 
        # Create DataFrame
        df_dist = pd.DataFrame(data)
        df_dist['Gausian Comonents']. astype(str)
        c=alt.Chart(df_dist).mark_bar().encode(
        x='Gausian Comonents',
        y='Avg_Mahalanobis_Distance',
        # The highlight will be set on the result of a conditional statement
        color=alt.condition(
        alt.datum.Avg_Mahalanobis_Distance == df_dist['Avg_Mahalanobis_Distance'].min(),  # If the year is 1810 this test returns True,
        alt.value('orange'),     # which sets the bar orange.
        alt.value('steelblue')   # And if it's not true it sets the bar steelblue.
        )
        ).properties(width=0)
        st.altair_chart(c, use_container_width=True)

 
        
        


        
        



        


















        # #if st.button('Click Here to To show performance Metrics'):
        #     Threshold_Hard_Member = st.slider('select the threshold value of propabity for defining hard and soft members?', 0.1, 0.99, 0.1)
        #     train_prob=gm.predict_proba(X_train).round(1)
        #     Hard_member=np.any(train_prob >Threshold_Hard_Member, axis=1)
        #     Hard_members=np.array(np.where(Hard_member==True)).size
        #     Total_Train=train_prob.shape[0]
        #     soft_members=Total_Train-Hard_members
        #     percent_hard_Members=(Hard_members/Total_Train)*100
        #     percent_hard_Members=(soft_members/Total_Train)*100
            
        #     #st.metric(label="silhouette score", value= metrics.silhouette_score(X_train,train_components))
        #     st.metric(label="% Hard Members", value= percent_hard_Members)
        #     st.metric(label="% Soft Members", value= soft_members)



















    



        
                

            