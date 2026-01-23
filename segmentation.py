import wfdb
import numpy
import numpy._core.numeric as _nx
from cvxEDA.src.cvxEDA import cvxEDA
import matplotlib.pyplot as pl
from scipy.stats import zscore
import csv
import neurokit2 as nk

EDA = [4]
FS = 8

def download_dataset():
        """
        Downloads the UTD dataset with only ACC, Temp and EDA data.
        """
        records = []
        for i in range(20):
                records.append("Subject" + str(i+1) +"_AccTempEDA")
        wfdb.dl_database(db_dir="noneeg", dl_dir="data/", records=records, annotators="all",overwrite=False)
        return True

def custom_split(ary:list, number_of_entries:int):
        """
        Splits an array into multiple arrays. 

        Please refer to the ''split'' documentation in the numpy library. The only difference
        between these functions is that this function splits an array into sub arrays with
        a given number of entries. If the last one array would be smaller than the given
        number, it is excluded from the new array.

        :param list: The list to be splitted.
        :param number_of_entries: The number of entries after which the split occurs.

        :return sub_arys: A numpy array with subarrays of the given length.
        """

        Ntotal = len(ary)
        Nsections = int(Ntotal/number_of_entries)
        if Nsections <=0:
                raise ValueError('number selection must be larger then 0.') from None
        section_sizes = number_of_entries
        
        #calculate points, where the array schould be split.
        div_points = []
        for i in range(Ntotal):
            div_points.append(i * section_sizes)

        #do the splitting
        sub_arys = []
        sary = _nx.swapaxes(ary,0,0)
        for i in range(Nsections):
                st = div_points[i]
                try:
                    end = div_points[i+1]
                    sub_arys.append(_nx.swapaxes(sary[st:end], 0, 0))
                except IndexError: #last division point could be out of bound
                    end = Ntotal -1 #do basically nothing

        return sub_arys

def extract_annotation(record:str, extension:str):
        """
        Extract annotation data from an WFDB annotation file.

        :param record: An WFDB annotation data file. Has to be a path if the file is not in the current directory.
        :param extension: The file type of the annotation file. Usually set to "atr".

        :return sample: The annotation marks as written in the annotation file.
        """

        annotation = wfdb.rdann(record, extension)
        return annotation.sample

def extract_signal(record:str, channel:list):
        """
        Extracts signal and annotations from a WFDB file.
        
        :param record: The name of the record to be read. Can be a path.
        :type record: str
        :param channel: Channels of the WFDB signal to be read. 
        :type channel: list

        :return signal: The extarcted signal.
        :return sampling_frequency: The sampling frequency of the extracted signal. 
        :return annotation_points: The extracted annotations from the annotation file of the signal.
        """
        signal,signal_data = wfdb.rdsamp(record_name=record, channels=channel,warn_empty=True)
        annotation_points = extract_annotation(record=record,extension="atr")
        sampling_frequency = signal_data["fs"]
        signal_length = signal_data["sig_len"]
        return signal, sampling_frequency, annotation_points

def components_separation(signal:list, method:str,fs=8):
        """
        Separates an EDA signal into its SCL(tonic) and SCR(phasic) components
        as shown in the cvxEDA paper.
        
        :param segment: A numpy array with the EDA signal.
        :param method: The method of component seperation to be used. Has to be "cvxEDA", 
        "smoothmedian', "highpass".
        :param fs: The sampling frequency of the signal. Defaults to 8Hz.

        :return components: Array of the phasic and tonic component.
        :return cleaned_signal: The cleaned EDA signal.
        :return scr_peaks: The detected SCR peaks.
        :return scr_onsets: The detected SCR onsets.
        :return recovery_time: The recovery times of the SCRs.
        """

        #normalize and separate the components
        if method not in ["cvxEDA", "smoothmedian", "highpass"]: raise ValueError("Method not recognized")
        signals, info = nk.eda_process(zscore(signal.ravel()), fs, method="neurokit", method_phasic=method, method_cleaning="neurokit", method_peaks="neurokit")
        components = []
        
        #extract other info for the feature vector
        components.append(signals["EDA_Tonic"].to_numpy())
        components.append(signals["EDA_Phasic"].to_numpy())
        cleaned_signal = signals["EDA_Clean"].to_numpy()
        scr_peaks = signals["SCR_Peaks"].to_numpy()
        scr_onsets = signals["SCR_Onsets"].to_numpy()
        recovery_time = signals["SCR_RiseTime"].to_numpy()


        return components, cleaned_signal, scr_peaks, scr_onsets, recovery_time

def segment_signal(record:str,channel:list, segment_length:int, method:str):
        """
        Segment a WFDB record of a given channel into given lengths in seconds.
        This function also separates the signal in the SCL,SCR components
        as seen in the cvxEDA module.

        :param record: A WFDB data file with the signal.
        :param channel: The channels which are to be considered. This attempt focuses on EDA, but there are more signals.
        :param segment_length: The length of each segment in seconds.
        :param method: The method of component seperation to be used. Has to be "cvxEDA", 
        "smoothmedian', "highpass".

        :return non_stress: A 2d numpy array with the non stress signal segments.
        :return stress: A 2d numpy array with the stress signal segments.
        :return phasic_non_stress: A 2d numpy array with the phasic component of non stress segments.
        :return phasic_stress: A 2d numpy array with the the phasic component of stress segments.
        :return tonic_non_stress: A 2d numpy array with the tonic component of non stress segments.
        :return tonic_stress: A 2d numpy array with the tonic component of stress segements.
        :return peaks_non_stress: A 2d numpy array with the SCR peaks of non stress segments.
        :return peaks_stress: A 2d numpy array with the SCR peaks of stress segments.
        :return onsets_non_stress: A 2d numpy array with the SCR onsets of non stress segments.
        :return onsets_stress: A 2d numpy array with the SCR onsets of stress segments.
        :return recovery_non_stress: A 2d numpy array with the SCR recovery times of non stress segments.
        :return recovery_stress: A 2d numpy array with the SCR recovery times of stress segments.
        """

        #read the signal and extract data
        signal, sampling_frequency, annotation_points = extract_signal(record, channel)
        

        #annotation constants for easier readability
        RELAX_ONE_START = annotation_points[0]
        PHYS_START = annotation_points[1]
        RELAX_TWO_START = annotation_points[2]
        COGN_START = annotation_points[3]
        RELAX_THREE_START = annotation_points[5]
        EMOT_START = annotation_points[6]
        RELAX_FOUR_START = annotation_points[7]

        #calculate the length of each segment
        entries_in_segment = int(sampling_frequency * segment_length)
        components, cleaned_signal, scr_peaks, scr_onsets, recovery_time = components_separation(signal,method, sampling_frequency)


        #initialize the arrays for the segments
        non_stress =[]
        stress =[]

        phasic_stress =[]
        phasic_non_stress =[]

        tonic_stress = []
        tonic_non_stress =[]

        peaks_non_stress =[]
        peaks_stress =[]

        onsets_non_stress =[]
        onsets_stress =[]
        
        recovery_non_stress =[]
        recovery_stress =[]



        #split the data using the custom splitting algorithm
        non_stress.extend(custom_split( cleaned_signal[RELAX_ONE_START : PHYS_START],entries_in_segment))
        non_stress.extend(custom_split( cleaned_signal[RELAX_TWO_START : COGN_START],entries_in_segment))
        stress.extend(custom_split( cleaned_signal[COGN_START : RELAX_THREE_START], entries_in_segment))
        non_stress.extend(custom_split( cleaned_signal[RELAX_THREE_START : EMOT_START], entries_in_segment))
        stress.extend(custom_split( cleaned_signal[EMOT_START : RELAX_FOUR_START], entries_in_segment))
        non_stress.extend(custom_split( cleaned_signal[RELAX_FOUR_START : ], entries_in_segment))

        phasic_non_stress.extend(custom_split( components[0][RELAX_ONE_START : PHYS_START],entries_in_segment))
        phasic_non_stress.extend(custom_split( components[0][RELAX_TWO_START : COGN_START],entries_in_segment))
        phasic_stress.extend(custom_split( components[0][COGN_START : RELAX_THREE_START], entries_in_segment))
        phasic_non_stress.extend(custom_split( components[0][RELAX_THREE_START : EMOT_START], entries_in_segment))
        phasic_stress.extend(custom_split( components[0][EMOT_START : RELAX_FOUR_START], entries_in_segment))
        phasic_non_stress.extend(custom_split( components[0][RELAX_FOUR_START : ], entries_in_segment))

        tonic_non_stress.extend(custom_split( components[1][RELAX_ONE_START : PHYS_START],entries_in_segment))
        tonic_non_stress.extend(custom_split( components[1][RELAX_TWO_START : COGN_START],entries_in_segment))
        tonic_stress.extend(custom_split( components[1][COGN_START : RELAX_THREE_START], entries_in_segment))
        tonic_non_stress.extend(custom_split( components[1][RELAX_THREE_START : EMOT_START], entries_in_segment))
        tonic_stress.extend(custom_split( components[1][EMOT_START : RELAX_FOUR_START], entries_in_segment))
        tonic_non_stress.extend(custom_split( components[1][RELAX_FOUR_START : ], entries_in_segment))  

        peaks_non_stress.extend(custom_split( scr_peaks[RELAX_ONE_START : PHYS_START],entries_in_segment))
        peaks_non_stress.extend(custom_split( scr_peaks[RELAX_TWO_START : COGN_START],entries_in_segment))
        peaks_stress.extend(custom_split( scr_peaks[COGN_START : RELAX_THREE_START], entries_in_segment))
        peaks_non_stress.extend(custom_split( scr_peaks[RELAX_THREE_START : EMOT_START], entries_in_segment))
        peaks_stress.extend(custom_split( scr_peaks[EMOT_START : RELAX_FOUR_START], entries_in_segment))
        peaks_non_stress.extend(custom_split( scr_peaks[RELAX_FOUR_START : ], entries_in_segment))  

        onsets_non_stress.extend(custom_split( scr_onsets[RELAX_ONE_START : PHYS_START],entries_in_segment))
        onsets_non_stress.extend(custom_split( scr_onsets[RELAX_TWO_START : COGN_START],entries_in_segment))
        onsets_stress.extend(custom_split( scr_onsets[COGN_START : RELAX_THREE_START], entries_in_segment))
        onsets_non_stress.extend(custom_split( scr_onsets[RELAX_THREE_START : EMOT_START], entries_in_segment))
        onsets_stress.extend(custom_split( scr_onsets[EMOT_START : RELAX_FOUR_START], entries_in_segment))
        onsets_non_stress.extend(custom_split( scr_onsets[RELAX_FOUR_START : ], entries_in_segment))  

        recovery_non_stress.extend(custom_split( recovery_time[RELAX_ONE_START : PHYS_START],entries_in_segment))
        recovery_non_stress.extend(custom_split( recovery_time[RELAX_TWO_START : COGN_START],entries_in_segment))
        recovery_stress.extend(custom_split( recovery_time[COGN_START : RELAX_THREE_START], entries_in_segment))
        recovery_non_stress.extend(custom_split( recovery_time[RELAX_THREE_START : EMOT_START], entries_in_segment))
        recovery_stress.extend(custom_split( recovery_time[EMOT_START : RELAX_FOUR_START], entries_in_segment))
        recovery_non_stress.extend(custom_split( recovery_time[RELAX_FOUR_START : ], entries_in_segment))  

             

        return non_stress, stress, phasic_non_stress, phasic_stress, tonic_non_stress, tonic_stress, onsets_non_stress, onsets_stress, peaks_non_stress, peaks_stress, recovery_non_stress, recovery_stress

def group_one_data(directory:str, channel:list, subject_number:int, segment_length:int, method:str):
        """
        Groups one subjects data into stress and non stress categories.
        Mostly a wrapper for the segment_signal function.

        :param directory: The directory in which the subject data is stored.
        :param channel: The channels which are to be considered. This attempt focuses on EDA, but there are more signals.
        :param subject_number: The number of the subject. In the case of the UTD dataset can be a number between 1 and 20.
        :param segment_length: The length of each segment in seconds.
        :param method: The method of component seperation to be used. Has to be "cvxEDA", 
        "smoothmedian', "highpass".

        For returns see segemnt_signal fucntion description.
        """

        #figure out the name of the subject file and extract the data
        record_name = directory + "/Subject" + str(subject_number) +"_AccTempEDA"
        return segment_signal(record=record_name, channel=channel, segment_length=segment_length, method=method)

def group_all_data_by_subject(directory:str, channel:list, data_count:int, segment_length:int, method:str):
        """
        Groups all data by subject into stress and non stress segments

        Defines a global array subject_data, through which the subject data can be accessed 
        by subject_data[subject_number][stress], in which stress is 1 and non stress 0.

        :param directory: The directory in which the subject data is stored.
        :param channel: The channels which are to be considered. This attempt focuses on EDA, but there are more signals.
        :param data_count: The number of subjects. 
        :param segment_length: The length of each segment in seconds.
        :param method: The method of component seperation to be used. Has to be "cvxEDA", 
        "smoothmedian', "highpass" or "sparseeda".

        returns:
        None
        """

        #declare needed arrays
        global subject_data
        subject_data = []
        temp_subject = []

        #go through the subject data and group it into stress and non stress by subject
        for i in range(data_count):
               temp_subject = group_one_data(directory, channel, i+1, segment_length, method)
               subject_data.append(temp_subject)

def get_subject_data(subject_number:int, type:int, segment_number=2048):
        """
        A Function to facilitate getting subject data.

        If the function encounters an IndexError, the whole stress/non stress normal/phasic/tonic
        category of the given subject is returned. 
        The subject_number is expected to be given from 1,...,n, e.g. to 
        access the first subject you have to use 1 as the subject_number
        (not 0, as you would normally as a programmer)

        type=0 - non stress segment, 
        type=1 - stress segment, 
        type=2 - phasic non stress segment, 
        type=3 - phasic stress segment, 
        type=4 - tonic non stress segment, 
        type=5 - tonic stress segment,
        type=6 - SCR onsets non stress segment,
        type=7 - SCR onsets stress segment,
        type=8 - SCR peaks non stress segment,
        type=9 - SCR peaks stress segment,
        type=10 - SCR recovery times non stress segment,
        type=11 - SCR recovery times stress segment

        :param subject_number: The number of the subject.
        :param type: The type of the segment, which will be returned. See the function description.
        :param segment_number: The number of the segment which will be reurned. Can be omitted, to 
        return all segments of the specified type of one subject.

        :return subject_data: specific segment or all segments of the specified type of one subject
        """

        try:
                return subject_data[subject_number - 1][type][segment_number-1]
        except IndexError:
                return subject_data[subject_number - 1][type]

def calculate_scr_features(phasic_segment:list, peaks:list, onsets:list, recovery_times:list ):
        """
        Calculate additional SCR features to form the feature vector. Used for the cvxEDA algorithm
        implementation. 

        Depreciated use the built-in neurokit eda_process function.

        For definitions of these features see:
        STRESS DETECTION THROUGH WRIST-BASED ELECTRODERMAL ACTIVITY MONITORING AND MACHINE LEARNING

        :param phasic_segment: An array with the phasic component of a segment.
        :param peaks: A list of indices of found peaks.
        :param onsets: A list of indices of found onsets.
        :param recovery_times: A list of recovery times.

        :return scr_onsets: The mean of all SCR onsets in the segment.
        :return scr_amplitudes: The mean of all SCR amplitudes in the segment.
        :return scr_recoveries: The mean of all SCR recovery times in the segment.
        """
        scr_recoveries = numpy.mean(recovery_times)
        scr_onsets = []
        scr_amplitudes = []
        for i in range(len(phasic_segment)):
                if onsets[i] == 1:
                        scr_onsets.append(phasic_segment[i])
                if peaks[i] ==1:
                        scr_amplitudes.append(phasic_segment[i])
        
        return numpy.mean(scr_onsets), numpy.mean(scr_amplitudes), scr_recoveries

def form_feature_vector(segment:list, phasic_segment:list, peaks:list, onsets:list, recovery_times:list):
        """
        Forms the Feature vector as seen in the paper.
        The both parameters should correspond to the same segment.

        STRESS DETECTION THROUGH WRIST-BASED ELECTRODERMAL ACTIVITY MONITORING AND MACHINE LEARNING

        :param segment: A numpy array with a segment of the signal.
        :param phasic_segment: A numpy array with the phasic component of a segment.  

        :return feature_vector: A feature vector as seen in the paper in the description.
        """

        scr_onsets, scr_amplitudes, scr_recoveries = calculate_scr_features(phasic_segment, peaks, onsets, recovery_times)
        return [segment.mean(), numpy.min(segment), numpy.max(segment), segment.std(), scr_onsets, scr_amplitudes, scr_recoveries]

def form_database(directory:str, channel:list, data_count:int, segment_length:int, method:str):
        """
        Forms a dictionary called database with feature vectors and some information about them. 
        database = [{subject, seg_mean, seg_min, seg_max, seg_std, rc_onsets, rc_amp, rc_rec, stress},...]

        :param directory: The directory in which the subject data is stored.
        :param channel: The channels which are to be considered. This attempt focuses on EDA, but there are more signals.
        :param data_count: The number of subjects. 
        :param segment_length: The length of each segment in seconds.
        :param method: The method of component seperation to be used. Has to be "cvxEDA", "smoothmedian', "highpass".

        Example:
        database = form_database(directory="data", channel=EDA, data_count=20, segment_length=30, method="cvxEDA")

        :return database: A database array with dictionaries as entries. See description above the 
        """
        #segment the data, group it and store it into an global array
        group_all_data_by_subject(directory, channel, data_count, segment_length, method)

        database = []
        #form the database
        for i in range(len(subject_data)):
                for k in range(2): #0 - non stress, 1 - stress
                        for l in range(len(subject_data[i][k])): #go through each segment

                                #form the feature vector and it to the dictionary
                                V = form_feature_vector(get_subject_data(i + 1, k,l + 1), get_subject_data(i + 1, k+2, l + 1), 
                                                        get_subject_data(i + 1, k+6, l + 1), get_subject_data(i + 1, k+8, l + 1), 
                                                        get_subject_data(i + 1, k+10, l + 1),)
                                subject = {'subject': i + 1, 'seg_mean': V[0], 
                                   'seg_min': V[1], 'seg_max': V[2], 'seg_std': V[3],
                                   'rc_onsets': V[4], 'rc_amp': V[5], 'rc_rec': V[6], 'stress': k}
                                database.append(subject)  
        return database
                             
def export_database(file_name:str ,dictionary:list):
        """
        Export a dictionary to a csv file. Creates a csvfile and stores it
        in the location of this file.

        :param file_name: The name under which the dictionary will be saved.
        :param dictionary: The dictionarz which will be exported as a csv file.
        

        returns:
        A csv file with all relevant infomation in the current directory.
        """
        with open(file_name, 'w', newline='') as csvfile:
                fieldnames = [ 'subject', 'seg_mean', 'seg_min', 'seg_max', 
                              'seg_std', 'rc_onsets', 'rc_amp', 'rc_rec', 'stress']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(dictionary)
        return True
       
def plot_segment(name:str,segment:list, phasic_segment:list, tonic_segment:list, title:str, fs=8):
        """
        Plots a given segment with the phasic components and saves it under a name as a svg file.

        :param name: The name under which the svg file will be saved.
        :param segment: A numpy array with a segment of the signal.
        :param phasic_segment: A numpy array with the phasic component of a segment. 
        :param tonic_segment: A numpy array with the tonic component of a segment. 
        :param title: The title of the plot.
        
        Examples:
        plot_segment("images/sm_segments.svg", get_subject_data(1,1,1), get_subject_data(1,3,1), get_subject_data(1,5,1),"EDA Signal Decomposition - Nonstress Segment - Smooth Median")
        plot_segment("results/non_stress_segment.svg", get_subject_data(1,0,4), get_subject_data(1,2,4), get_subject_data(1,4,4),"EDA Signal Decomposition - Nonstress Segment")
        plot_segment("results/stress_segment.svg", get_subject_data(1,1,4), get_subject_data(1,3,4), get_subject_data(1,5,4),"EDA Signal Decomposition - Stress Segment") 

        plot_segment("results/non_stress_segment_alt.svg", get_subject_data(1,0,16), get_subject_data(1,2,16), get_subject_data(1,4,16),"EDA Signal Decomposition - Nonstress Segment")
        plot_segment("results/stress_segment_alt.svg", get_subject_data(1,1,16), get_subject_data(1,3,16), get_subject_data(1,5,16),"EDA Signal Decomposition - Stress Segment") 

        returns:
        A visual representation of the segment and a svg file with the representation.
        """
        tm = numpy.arange(0, len(segment)) /fs
        pl.plot(tm, segment, color='k', label="Segment") #black
        pl.plot(tm, phasic_segment, color='r', label="Phasic component") #red
        pl.plot(tm, tonic_segment, color='b', label="Tonic component") #blue

        pl.title(title)
        pl.xlabel("Time (s)")
        pl.ylabel("Amplitude")
        pl.legend(loc="lower right")
        pl.savefig(name, format='svg',dpi=300, bbox_inches='tight')
        pl.show()
        

        return True
