from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import pickle
import pandas as pd

if __name__ == '__main__':
  label_encoder = pickle.load(open('./models/encoder.pkl', 'rb'))
  model = pickle.load(open('./models/model.pkl', 'rb'))
  while True:
    age = int(input('Age: '))
    gender = input('Gender: ')
    education = input('Education Level: ')
    job = input('Job Title: ')
    experience = int(input('Years of Experience: '))

    gender, education, job = map(lambda e: label_encoder.transform([e])[0], (gender, education, job))

    data = pd.DataFrame({
      'Age': [age],
      'Gender': [gender],
      'Education Level': [education],
      'Job Title': [job],
      'Years of Experience': [experience]
    })
    prediction = model.predict(data)
    answer = round(prediction[0])
    print(f'Predicted Salary: {answer}')
