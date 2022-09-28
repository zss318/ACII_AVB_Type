
%% Settings
path_in = ''; % path to raw audio files

path_save = ''; % path to processed audio files
mkdir( path_save )

fs = 16e3;

windowDuration = 0.2;
numWindowSamples = round(windowDuration*fs);
win = hamming(numWindowSamples,'periodic');

percentOverlap = 80;
overlap = round(numWindowSamples*percentOverlap/100);

mergeDuration = 0.2;
mergeDist = round(mergeDuration*fs);

%% Start
cd(path_in)
List = dir(path_in);

for aa = 3:length(List)
    
    file_in = List(aa).name;
    
    [~,filename,~] = fileparts(file_in);
    file_save = [path_save '/' filename '.wav'];


    [audio_in,fs] = audioread( file_in);


    [speechIndices,thresholds] = detectSpeech(audio_in,fs,"Window",win,"OverlapLength",overlap,"MergeDistance",mergeDist);


    
    [r,c] = size( speechIndices );
    if r>1

        SigOut = [];
        for bb = 1:r
            audio_segment = audio_in(speechIndices(bb,1):speechIndices(bb, 2));
            audio_segment_norm = rescale(audio_segment,-1,1);
            SigOut = [SigOut;audio_segment_norm];
        end

    % Incase there is only one segment    
    else

        audio_segment = audio_in(speechIndices(1,1):speechIndices(1, 2));
        SigOut = rescale(audio_segment,-1,1);

    end

    % -- Write audio file
    audiowrite( file_save, SigOut, fs)


end