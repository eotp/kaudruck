from skimage import io
import numpy as np
import matplotlib.pyplot as plt
import json

class KauDruck():
    def __init__(self, image_path, aoi=None, aor=None, threshold=0.3):
        self.img = self.load_img(image_path)
        self.force_correction_table_path = '../models/force_correction.json' 
        self.force_correction_data = self._load_force_correction_table()
        self.force_correction_model = self._fit_force_correction()
        
        self.img = self.img / 255   # scale image between 0 and 1
        self.area_corr_fact = 1.0021
        self.force_aor = 50         # Newton
        if aoi is None:
            self.aoi = self.img
            
        else:
            self.aoi, self.aoi_bb = self.crop_img(self.img, 
                                                  xy=aoi[0], 
                                                  height=aoi[1],
                                                  width=aoi[2])
        self.aor = None
        self.threshold = threshold
        
    def load_img(self, path_to_file):
        img = io.imread(path_to_file)
        return img
    
    def _load_force_correction_table(self):
        with open(self.force_correction_table_path) as f:
            force_correction_data = json.load(f)
        return force_correction_data
    def _fit_force_correction(self):
        x = self.force_correction_data['x_computed']
        y = self.force_correction_data['y_measured']
        z = np.polyfit(x, y, 2)
        p = np.poly1d(z)
        return p

    def crop_img(self, img, xy, height, width):
        img_cropped = img[xy[1]:xy[1]+height, xy[0]:xy[0]+width]
        
        bb_left = ((xy[0], xy[0]), (xy[1], xy[1]+height))
        bb_upper = ((xy[0], xy[0]+width), (xy[1], xy[1]))
        bb_right = ((xy[0]+width, xy[0]+width), (xy[1], xy[1]+height))
        bb_lower = ( (xy[0], xy[0]+width), (xy[1]+height, xy[1]+height))
        return img_cropped, (bb_left, bb_upper, bb_right, bb_lower)
        
    
    def plot(self, ax=None, *args, **kwargs):
        
        if ax is None:
            fig, ax = plt.subplots()
        ax.imshow(self.img, *args, **kwargs)
        
        if self.aoi is not None:
            params={'color'     : 'green' , 
                    'linewidth' : 2, 
                    'linestyle' : 'solid'}
            for i in range(len(self.aoi_bb)):
                ax.plot(self.aoi_bb[i][0], 
                         self.aoi_bb[i][1], 
                         color=params['color'], 
                         linewidth=params['linewidth'], 
                         linestyle=params['linestyle'])
        
        if self.aor is not None:
            params={'color'     : 'black' , 
                    'linewidth' : 2, 
                    'linestyle' : 'dashed'}         
            for i in range(len(self.aor_bb)):
                ax.plot(self.aor_bb[i][0],
                         self.aor_bb[i][1], 
                         color=params['color'], 
                         linewidth=params['linewidth'], 
                         linestyle=params['linestyle'])
    
    def set_aoi(self, xy, height, width):
        self.aoi, self.aoi_bb = self.crop_img(self.img, xy, height, width)
    
    def set_aor(self, xy, height, width):
        self.aor, self.aor_bb = self.crop_img(self.img, xy, height, width)
    
    def set_schwelle(self, threshold):
        self.threshold = threshold
    
    def compute_results(self, aoi):
        intensity = self.compute_intensitat(aoi)
        m = intensity < self.threshold
        intensity[m] = 0 
        
        flaeche = sum(sum(intensity > 0))
        farbgewicht = sum(sum(intensity))
        quotient = farbgewicht / flaeche
  
        return flaeche, farbgewicht, quotient

    def compute_kaudruck(self, weight_aoi, weight_aor, force_aor):
        return (weight_aoi / weight_aor) * force_aor
    
    def compute_intensitat(self, aoi):
        green = aoi[:,:,1].copy()
        blue = aoi[:,:,2].copy()
        intensity = ( (1-green) + (1-blue) ) / 2
        return intensity
    
    def set_area_corr_fact(self, corr_fact):
        self.area_corr_fact = corr_fact
    
    def set_aor_force(self, force_N):
        self.force_aor = force_N
    
    def compute_area_mm(self, area_pixel):
        area_mm = ((25.4/800)**2 * area_pixel) / self.area_corr_fact
        return area_mm
    
    def compute_pressure(self, force, area):
        return force / area
    
    def run_analysis(self):
        
        aor_flaeche, aor_farbgewicht, aor_quotient = self.compute_results(aoi=self.aor)
        self.aor_area = aor_flaeche
        self.aor_color_weight = aor_farbgewicht
        self.aor_color_weight_area_ratio = aor_quotient 
        
        area_aor_corrected = self.compute_area_mm(self.aor_area)
        self.aor_pressure = self.compute_pressure(force=self.force_aor, 
                                                   area=area_aor_corrected)
        
        print(f'\nArea of reference')
        print(f'------------------------------')
        print(f'Fläche (pixel):          {int(np.round(self.aor_area))}')
        print(f'Farbgewicht:             {int(np.round(self.aor_color_weight))}')
        print(f'Gewicht/Fläche:          {np.round(self.aor_color_weight_area_ratio,3)}')
        
        print(f'Fläche (mm, korrigiert): {np.round(area_aor_corrected,2)}')
        print(f'Kraft (N):               {np.round(self.force_aor,2)}')
        print(f'Druck (MPa):             {np.round(self.aor_pressure,2)}')
        
        
        aoi_flaeche, aoi_farbgewicht, aoi_quotient = self.compute_results(aoi=self.aoi)
        self.aoi_area = aoi_flaeche
        self.aoi_color_weight = aoi_farbgewicht
        self.aoi_color_weight_area_ratio = aoi_quotient   
        
        area_aoi_corrected = self.compute_area_mm(self.aoi_area)
        
        self.force_aoi = self.compute_kaudruck(weight_aoi=self.aoi_color_weight,
                                               weight_aor=self.aor_color_weight, 
                                               force_aor=self.force_aor)
        
        self.aoi_pressure = self.compute_pressure(force=self.force_aoi, 
                                                   area=area_aoi_corrected)
        
        self.force_aoi_corrected =  self.force_correction_model(self.force_aoi)
        
        self.aoi_pressure_corrected = self.compute_pressure(force=self.force_aoi_corrected,
                                                            area=area_aoi_corrected)
        

        print(f'\nArea of interest')
        print(f'------------------------------')
        print(f'Fläche (pixel):          {int(np.round(self.aoi_area))}')
        print(f'Farbgewicht:             {int(np.round(self.aoi_color_weight))}')
        print(f'Gewicht/Fläche:          {np.round(self.aoi_color_weight_area_ratio,3)}')
        
        print(f'Fläche (mm, korrigiert): {np.round(area_aoi_corrected,2)}')
        print(f'Kraft (N):               {np.round(self.force_aoi,2)}')
        print(f'Druck (MPa):             {np.round(self.aoi_pressure,2)}')
        print(f'Kraft (N, korrigiert):   {np.round(self.force_aoi_corrected,2)}')
        print(f'Druck (MPa, korrigiert): {np.round(self.aoi_pressure_corrected,2)}')

         
def plot_aoi_aor(kd):
    fig, ax = plt.subplots(ncols=3, figsize=(18,8))
    kd.plot(ax=ax[0])
    ax[1].imshow(kd.aoi)
    ax[2].imshow(kd.aor)
    ax[0].set_title('Sample image', size=18)
    ax[1].set_title('Area of interest (AOI)', size=18)
    ax[2].set_title('Area of reference (AOR)', size=18)
    fig.tight_layout()