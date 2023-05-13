# Min Max Scaler Class
# Containes initialization, fit, transform and inverse transform functions
# The class object saves the data over which it was fit to keep track of when doing Inverse Transform

class MinMaxScaler:
    
    def __init__(self,min_range,max_range):
        self.min_range = min_range
        self.max_range = max_range
        self.d_set_obj = {}
        
    def fit(self,d_set):
        d_set_t = list(zip(*d_set))
        for row in d_set_t:
            self.d_set_obj[d_set_t.index(row)] = [min(row),max(row)]
        return 1
    
    def transform(self,d_set):
        if len(self.d_set_obj) == 0:
            raise ValueError("Missing fit() before transform()")
        else:
            d_set_t = list(zip(*d_set))
            n_scaled = []
            for row in d_set_t:
                row_scaled = []
                for value in list(row):
                    n_std = (value - self.d_set_obj[d_set_t.index(row)][0]) / (self.d_set_obj[d_set_t.index(row)][1] - self.d_set_obj[d_set_t.index(row)][0])
                    row_scaled.append(n_std * (self.max_range-(self.min_range)) + (self.min_range))
                n_scaled.append(row_scaled)
            return [list(x) for x in zip(*n_scaled)]
        
    def inverse_transform(self,d_set):
        if len(self.d_set_obj) == 0:
            raise ValueError("Missing fit() before inverse_transform()")
        else:
            d_set_t = list(zip(*d_set))
            n_orig = []
            for row in d_set_t:
                row_scaled = []
                for value in list(row):
                    n_std = (value - self.min_range)/(self.max_range - self.min_range)
                    row_scaled.append(n_std*(self.d_set_obj[d_set_t.index(row)][1]-self.d_set_obj[d_set_t.index(row)][0]) + self.d_set_obj[d_set_t.index(row)][0])
                n_orig.append(row_scaled)
            return [list(x) for x in zip(*n_orig)]
