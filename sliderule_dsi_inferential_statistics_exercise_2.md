
# Examining Racial Discrimination in the US Job Market

### Background
Racial discrimination continues to be pervasive in cultures throughout the world. Researchers examined the level of racial discrimination in the United States labor market by randomly assigning identical résumés to black-sounding or white-sounding names and observing the impact on requests for interviews from employers.

### Data
In the dataset provided, each row represents a resume. The 'race' column has two values, 'b' and 'w', indicating black-sounding and white-sounding. The column 'call' has two values, 1 and 0, indicating whether the resume received a call from employers or not.

Note that the 'b' and 'w' values in race are assigned randomly to the resumes when presented to the employer.

<div class="span5 alert alert-info">
### Exercises
You will perform a statistical analysis to establish whether race has a significant impact on the rate of callbacks for resumes.

Answer the following questions **in this notebook below and submit to your Github account**. 

   1. What test is appropriate for this problem? Does CLT apply?
   2. What are the null and alternate hypotheses?
   3. Compute margin of error, confidence interval, and p-value.
   4. Write a story describing the statistical significance in the context or the original problem.
   5. Does your analysis mean that race/name is the most important factor in callback success? Why or why not? If not, how would you amend your analysis?

You can include written notes in notebook cells using Markdown: 
   - In the control panel at the top, choose Cell > Cell Type > Markdown
   - Markdown syntax: http://nestacms.com/docs/creating-content/markdown-cheat-sheet


#### Resources
+ Experiment information and data source: http://www.povertyactionlab.org/evaluation/discrimination-job-market-united-states
+ Scipy statistical methods: http://docs.scipy.org/doc/scipy/reference/stats.html 
+ Markdown syntax: http://nestacms.com/docs/creating-content/markdown-cheat-sheet
</div>
****


```python
import pandas as pd
import numpy as np
from scipy import stats
```


```python
data = pd.io.stata.read_stata('/Users/jason/svn/springboard/racial_disc/data/us_job_market_discrimination.dta')
```


```python
# number of callbacks for black-sounding names
print sum(data[data.race=='b'].call)
```

    157.0



```python
data.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>ad</th>
      <th>education</th>
      <th>ofjobs</th>
      <th>yearsexp</th>
      <th>honors</th>
      <th>volunteer</th>
      <th>military</th>
      <th>empholes</th>
      <th>occupspecific</th>
      <th>...</th>
      <th>compreq</th>
      <th>orgreq</th>
      <th>manuf</th>
      <th>transcom</th>
      <th>bankreal</th>
      <th>trade</th>
      <th>busservice</th>
      <th>othservice</th>
      <th>missind</th>
      <th>ownership</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>b</td>
      <td>1</td>
      <td>4</td>
      <td>2</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>17</td>
      <td>...</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td></td>
    </tr>
    <tr>
      <th>1</th>
      <td>b</td>
      <td>1</td>
      <td>3</td>
      <td>3</td>
      <td>6</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>316</td>
      <td>...</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td></td>
    </tr>
    <tr>
      <th>2</th>
      <td>b</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>19</td>
      <td>...</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td></td>
    </tr>
    <tr>
      <th>3</th>
      <td>b</td>
      <td>1</td>
      <td>3</td>
      <td>4</td>
      <td>6</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>313</td>
      <td>...</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td></td>
    </tr>
    <tr>
      <th>4</th>
      <td>b</td>
      <td>1</td>
      <td>3</td>
      <td>3</td>
      <td>22</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>313</td>
      <td>...</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>Nonprofit</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 65 columns</p>
</div>



## Hypothesis tests
The null hypothesis, $H_0$, is that there is no difference betwen the proportion of call backs given to the group with black sounding names compared to the other fictional candidates. The alternate hypothesis, $H_1$, given the historical context, is that resumes with black sounding names will get fewer call backs. We have 2835 observations for each group, which we expect to follow a Bernoulli distribution. Therefore, the CLT should apply to this case.


```python
import scipy as sp
class callbacks(object):
    
    def __init__(self, calls):
        self.calls = calls
        self.p = -1
        self.variance = -1
        self.standard_deviation = -1
        self.nobs = len(calls)
        
    def mean(self):
        self.p = np.mean(self.calls)
        
    def var(self):
        if self.p < 0:
            self.mean()
            
        self.variance = self.p * (1 - self.p)
        
    def sd(self):
        if self.variance < 0:
            self.var()
            
        self.standard_deviation = np.sqrt(self.variance)
        
    def ci(self, alpha):
        # divided by 2 so we center our interval
        n_sds = sp.stats.norm.ppf(alpha / 2)
        tmp = np.sqrt(self.p * (1 - self.p) / self.nobs)
        return [self.p + n_sds * tmp, self.p - n_sds * tmp]
        
    def z_score_mean(self,other_mean):
        if self.p < 0:
            self.mean()
        return (other_mean - self.p) / (np.sqrt(self.p * (1 - self.p) / self.nobs))
        
    def __str__(self):
        if self.standard_deviation < 0:
            self.sd()
            
        return "Number of observations: " + str(self.nobs)\
               + "\n" + "Mean = " + str(np.round(self.p,3)) + "\n" + "Variance = " + str(np.round(self.variance, 3))\
               + "\n" + "Standard deviation = " + str(np.round(self.standard_deviation, 3))
```

### Descriptive statistics of the groups


```python
b_calls = callbacks(data.call[data.race =='b'])
print("Black sounding names:")
print(b_calls)
print("\n")

other_calls = callbacks(data.call[data.race !='b'])
print("Other names:")
print(other_calls)
print("\n")

all_calls = callbacks(data.call)
print("All names:")
print(all_calls)
```

    Black sounding names:
    Number of observations: 2435
    Mean = 0.064
    Variance = 0.06
    Standard deviation = 0.246
    
    
    Other names:
    Number of observations: 2435
    Mean = 0.097
    Variance = 0.087
    Standard deviation = 0.295
    
    
    All names:
    Number of observations: 4870
    Mean = 0.08
    Variance = 0.074
    Standard deviation = 0.272


### Compute z-score and p-value


```python
z_score = all_calls.z_score_mean(b_calls.p)
print z_score
```

    -4.10841310411



```python
alpha = 0.05

z_crit = sp.stats.norm.ppf(alpha)
p_stat = sp.stats.norm.cdf(z_score)
if z_score < z_crit:
    print("We found evidence to reject the null hypothesis:")
else:
    print("We did not find evidence to reject the null hypothesis:")
    
print("\tz crit   : " + str(z_crit))
print("\tz score  : " + str(z_score))
print("\talpha    : " + str(alpha))
print("\tp value  : " + str(p_stat))
```

    We found evidence to reject the null hypothesis:
    	z crit   : -1.64485362695
    	z score  : -4.10841310411
    	alpha    : 0.05
    	p value  : 1.99193521228e-05


### Confidence interval
Note that the mean percentage of calls for the entire group is outside the 95% confidence interval for the fictional applicants with black sounding names.


```python
print("95% confidence interval for percenage of calls to applicants with black sounding names is: \n"\
      + str(b_calls.ci(0.05)))
```

    95% confidence interval for percenage of calls to applicants with black sounding names is: 
    [0.054721406960874547, 0.074231364464372646]


### Margin of error
Below is our estimate of the margin of error for our estimate of the percentage of calls to applicants with black sounding names.


```python
print("+/- " + str(np.abs(np.round(100 * (b_calls.ci(0.05)[0] - b_calls.p) / b_calls.p))) + "%")
```

    +/- 15.0%


### Summary of findings
Our analysis of the data provide strong evidence that applicants with names perceived to be called back for interviews less often than others. The p value for this difference was 2e-5, which means that if there was no difference, we would expect to find an effect this strong or stronger approximately 1 out of every 50,000 samples. So while there may be othe problems associated with this experiment, we feel very confident that the result is not due to sampling variability

Our best estimate of the difference is the difference between the means of the samples with and without a name that is perceived to represent and african american. The sample means are 0.08 and 0.064. So we estimate that an african american sounding name will decrease your likelihood of receiving an call back for an interview by approximately 20%.

### Is a name perceived to be associated with african american applicants the most significant factor in receiving a callback?

Our analysis only assessed whether the evidence presented supported the hypothesis that applicants with african american sounding names would receive fewer calls for interviews. We could use logistic regression to assess the effect sizes of the different variables relevant to this problem.


```python

```
