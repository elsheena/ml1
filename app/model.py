import pandas as pd
from catboost import CatBoostClassifier
import os
from sklearn.impute import KNNImputer
def log(message):
    print(f"[LOG] {message}")

class SpaceTitanicModel:
    def __init__(self):
        self.default_model_path = 'models/catboost_model_best.cbm'
        # Initialize empty model if file doesn't exist
        if os.path.exists(self.default_model_path):
            self.models = CatBoostClassifier().load_model(self.default_model_path)
        else:
            self.models = CatBoostClassifier()

    def preprocess(self, test:pd.DataFrame) -> pd.DataFrame:
        train = pd.read_csv('./data/train.csv')
        train.drop('Transported',axis=1,inplace=True)
        WITH=pd.concat([train,test])
        WITH.drop(['PassengerId','Cabin','Name'],axis=1,inplace=True)
        WITH.replace({'VIP' : {False : 0, True : 1}},inplace=True)
        WITH.replace({'CryoSleep' : {False : 0, True : 1}},inplace=True)
        WITH.replace({'HomePlanet' : {'Europa' : 0, 'Earth' : 1,'Mars': 2}},inplace=True)
        WITH.replace({'Destination' : {'TRAPPIST-1e' : 0, 'PSO J318.5-22' : 1,'55 Cancri e': 2}},inplace=True)

        imputer = KNNImputer(n_neighbors=1, weights="uniform")

        l=imputer.fit_transform(WITH)

        WITH1=pd.DataFrame(l,columns=WITH.columns)
        ind=range(12970)
        WITH1['Index']=ind
        WITH1=WITH1.set_index('Index')

        Home_planet=pd.get_dummies(WITH1.HomePlanet).add_prefix('HomePlanet')
        WITH1=WITH1.merge(Home_planet,on='Index')
        WITH1=WITH1.drop(['HomePlanet'],axis=1)

        Destination=pd.get_dummies(WITH1.Destination).add_prefix('Destination')
        WITH1=WITH1.merge(Destination,on='Index')
        WITH1=WITH1.drop(['Destination'],axis=1)

        test1=WITH1[8693:]

        return test1
        
        

    def train(self, 
              data_path, 
              target_column, 
              model_path=None,
              iterations=1000,
              learning_rate=0.1,
              depth=6):
        try:
            # Use data_path parameter instead of hardcoded path
            df = pd.read_csv(data_path)
        
            # Separate features and target
            X = self.preprocess(df.drop(target_column, axis=1))
            y = df[target_column]
        
            # Initialize and train model
            self.models = CatBoostClassifier(
                iterations=iterations,
                learning_rate=learning_rate,
                depth=depth,
                verbose=100  # Print training progress every 100 iterations
        )
        
            self.models.fit(X, y)
        
            # Save the model if path is provided
            if model_path:
                self.models.save_model('../models/catboost_model_best.cbm')

            return self.models
        
        except Exception as e:
            log(f"Error during training: {str(e)}")
            raise

    def predict(self, 
                data_path, 
                model_path=None,
                output_path='predictions.csv'):
        try:
            # Load and prepare data
            df = pd.read_csv(data_path)
            passenger_ids = df['PassengerId']  # Save PassengerId before preprocessing
            
            df = self.preprocess(df)
            
            # Make predictions
            predictions = self.models.predict(df)
            
            # Create submission dataframe
            submission = pd.DataFrame({
                'PassengerId': passenger_ids,
                'Transported': predictions
            })
            
            # Save predictions
            submission.to_csv('../predictions/predictions.csv', index=False)
            log(f"Predictions saved to ../predictions/predictions.csv")
            
        except Exception as e:
            log(f"Error during prediction: {str(e)}")