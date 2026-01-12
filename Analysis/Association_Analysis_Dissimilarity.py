import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

data = pd.read_excel('Source Data File.xlsx')


increased = data['Overall_Increased_Similarity']
dissimilarity = data['Overall_Microbiome_Dissimilarity']


df = pd.DataFrame({
    'Increased': increased,
    'Dissimilarity': dissimilarity
}).dropna()


corr_coef, p_value = stats.pearsonr(df['Increased'], df['Dissimilarity'])


plt.figure(figsize=(8, 6))


df['Color'] = np.where(df['Increased'] > 0, 'Positive', 'Negative')


sns.scatterplot(data=df, x='Increased', y='Dissimilarity', 
                hue='Color', palette={'Positive': 'blue', 'Negative': 'red'},
                s=40, alpha=0.7)


sns.regplot(data=df, x='Increased', y='Dissimilarity', 
            scatter=False, color='black', line_kws={'linestyle': '--'})


plt.text(0.05, 0.95, f'R = {corr_coef:.2f}\np = {p_value:.3f}',
         transform=plt.gca().transAxes,
         fontsize=10, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.xlabel('Increased Similarity')
plt.ylabel('Profile Dissimilarity')
plt.title('Association Analysis')
plt.legend(title='Increased Similarity')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
