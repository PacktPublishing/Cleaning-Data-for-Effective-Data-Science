import io
import pandas as pd
import warnings
warnings.simplefilter('ignore')

data = """
Student#,Last Name,First Name,Favorite Color,Age
1,Johnson,Mia,periwinkle,12
2,Lopez,Liam,blue,green,13
3,Lee,Isabella,,11
4,Fisher,Mason,gray,-1
5,Gupta,Olivia,9,102
6,,Robinson,,Sophia,,blue,,12
"""

clean = """
Student_No,Last_Name,First_Name,Favorite_Color,Age
1,Johnson,Mia,periwinkle,12
2,Lopez,Liam,blue-green,13
3,Lee,Isabella,<missing>,11
4,Fisher,Mason,gray,N/A
5,Gupta,Olivia,sepia,N/A
6,Robinson,Sophia,blue,12
"""

cleaned = pd.read_csv(io.StringIO(clean)).set_index('Student_No')