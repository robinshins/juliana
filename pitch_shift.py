import librosa
import os
import subprocess
import numpy as np
import scipy.io.wavfile as wavfile
import torch
import soundfile as sf
import torch.nn as nn
import pyworld as pw



# Wave-U-Net 모델 정의
class WaveUNet(nn.Module):
    def __init__(self, num_channels=1, num_outputs=1, num_layers=6, features_root=16, kernel_size=5):
        super(WaveUNet, self).__init__()
        self.num_layers = num_layers
        self.downsampling_layers = nn.ModuleList()
        self.upsampling_layers = nn.ModuleList()
        
        in_channels = num_channels
        out_channels = features_root
        
        # 다운샘플링 레이어
        for i in range(num_layers):
            conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=kernel_size//2)
            self.downsampling_layers.append(nn.Sequential(conv, nn.LeakyReLU(0.2)))
            in_channels = out_channels
            out_channels = min(out_channels * 2, 512)  # 최대 채널 수를 512로 제한
        
        # 업샘플링 레이어
        for i in range(num_layers):
            conv = nn.ConvTranspose1d(in_channels * 2, in_channels // 2, kernel_size=kernel_size, stride=1, padding=kernel_size//2)
            self.upsampling_layers.append(nn.Sequential(conv, nn.LeakyReLU(0.2)))
            in_channels = in_channels // 2
        
        # 최종 출력 레이어
        self.final_conv = nn.Conv1d(in_channels, num_outputs, kernel_size=1)
        
    def forward(self, x):
        # 다운샘플링
        skip_connections = []
        for down in self.downsampling_layers:
            x = down(x)
            skip_connections.append(x)
            x = nn.functional.avg_pool1d(x, 2)
        
        # 업샘플링
        for up, skip in zip(self.upsampling_layers, reversed(skip_connections)):
            x = nn.functional.interpolate(x, scale_factor=2, mode='linear', align_corners=False)
            # 크기 조정
            if x.size(2) > skip.size(2):
                x = x[:, :, :skip.size(2)]
            elif x.size(2) < skip.size(2):
                skip = skip[:, :, :x.size(2)]
            x = torch.cat((x, skip), dim=1)
            x = up(x)
        
        # 최종 출력
        x = self.final_conv(x)
        return x

def get_chord_intervals(scale_type='major'):
    """
    스케일 타입에 따른 화음 간격 반환
    
    Args:
        scale_type: 'major' 또는 'minor'
    Returns:
        dict: 각 화음 구성에 필요한 음정 간격
    """
    if scale_type == 'major':
        return {
            'I':   [0, 4, 7],      # 으뜸화음 (major)
            'ii':  [-10, -7, -3],  # 두번째 화음 (minor)
            'iii': [-8, -5, -1],   # 세번째 화음 (minor)
            'IV':  [-7, -3, 0],    # 네번째 화음 (major)
            'V':   [-5, -1, 2],    # 다섯번째 화음 (major)
            'vi':  [-3, 0, 4],     # 여섯번째 화음 (minor)
            'vii': [-1, 2, 5],     # 일곱번째 화음 (diminished)
            
            # 7화음
            'I7':   [0, 4, 7, 11],     # 으뜸화음7 (major7)
            'V7':   [-5, -1, 2, 5],    # 딸림7화음 (dominant7)
        }
    else:  # minor
        return {
            'i':   [0, 3, 7],      # 으뜸화음 (minor)
            'ii':  [-10, -7, -3],  # 두번째 화음 (diminished)
            'III': [-8, -5, -1],   # 세번째 화음 (major)
            'iv':  [-7, -4, 0],    # 네번째 화음 (minor)
            'v':   [-5, -2, 2],    # 다섯번째 화음 (minor)
            'VI':  [-3, 0, 4],     # 여섯번째 화음 (major)
            'VII': [-1, 2, 6],     # 일곱번째 화음 (major)
            
            # 7화음
            'i7':   [0, 3, 7, 10],     # 으뜸화음7 (minor7)
            'V7':   [-5, -1, 2, 5],    # 딸림7화음 (dominant7)
        }

def pitch_shift_chord(input_path, output_dir, scale_type='major', chord_name='I'):
    os.makedirs(output_dir, exist_ok=True)
    
    # chord_intervals 가져오기 추가
    chord_intervals = get_chord_intervals(scale_type)
    
    for semitones in chord_intervals[chord_name]:
        # 각 음정마다 새로운 더블링 생성
        random_seed = np.random.randint(0, 10000)
        torch.manual_seed(random_seed)
        
        # 임시 더블링 파일 경로
        doubled_path = os.path.join(output_dir, f"doubled_temp_{semitones}.wav")
        
        # WaveUNet으로 더블링 생성
        model = WaveUNet().to('cpu')
        model.load_state_dict(torch.load("./vocal_waveunet_model_all_songs.pth", map_location='cpu'))
        model.eval()
        
        # 오디오 로드 및 더블링 처리
        original, sr = librosa.load(input_path, sr=None)
        original_tensor = torch.from_numpy(original).float().unsqueeze(0).unsqueeze(0)
        
        with torch.no_grad():
            modified = model(original_tensor).squeeze().numpy()
        
        # 위상 처리
        n_fft = 2048
        hop_length = 512
        stft_original = librosa.stft(original, n_fft=n_fft, hop_length=hop_length)
        mag_original, phase_original = librosa.magphase(stft_original)
        
        stft_modified = librosa.stft(modified, n_fft=n_fft, hop_length=hop_length)
        mag_modified, phase_modified = librosa.magphase(stft_modified)
        
        mag_result = mag_original
        phase_result = phase_modified
        
        result = librosa.istft(mag_result * phase_result, hop_length=hop_length)
        
        # 시간 지연 추가 (10ms)
        delay_samples = int(0.01 * sr)
        result_delayed = np.pad(result, (delay_samples, 0), mode='constant')[:len(result)]
        
        # 더블링된 결과 저장
        sf.write(doubled_path, result_delayed, sr)
        
        # 생성된 더블링에 대해 피치 시프트 적용
        base_filename = os.path.splitext(os.path.basename(input_path))[0]
        output_name = f"{base_filename}_doubled_{scale_type}_{chord_name}_{'+' if semitones >= 0 else ''}{semitones}.wav"
        output_path = os.path.join(output_dir, output_name)
        
        # soundstretch로 피치 시프트
        cmd = ['soundstretch', doubled_path, output_path, '-pitch=' + str(semitones), '-tempo=0', '-rate=0', '-speech', '-sequence=30', '-naa']
        
        try:
            subprocess.run(cmd, check=True)
            y, sr = librosa.load(output_path, sr=None)
            y = librosa.util.normalize(y)
            y_int = np.int16(y * 32767)
            wavfile.write(output_path, sr, y_int)
            print(f'생성 완료: {output_name} (피치 시프트: {semitones})')
            
        except Exception as e:
            if os.path.exists(output_path):
                os.remove(output_path)
            raise Exception(f"처리 실패: {str(e)}")
        
        # 임시 파일 삭제
        os.remove(doubled_path)


def make_doubling(input_path, output_path):
    """
    입력된 WAV 파일의 더블링을 생성하는 함수
    
    Args:
        input_path: 입력 WAV 파일 경로
        output_path: 출력 WAV 파일 경로
    """
    model = WaveUNet().to('cpu')
    model.load_state_dict(torch.load("./vocal_waveunet_model_all_songs.pth", map_location='cpu', weights_only=True))
    model.eval()
    
    # 1. 오디오 로드
    original, sr = librosa.load(input_path, sr=None)
    
    # 2. Wave-U-Net을 통한 위상 변경
    original_tensor = torch.from_numpy(original).float().unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        modified = model(original_tensor).squeeze().numpy()
    
    # 3. 원본 포먼트 추출
    n_fft = 2048
    hop_length = 512
    stft_original = librosa.stft(original, n_fft=n_fft, hop_length=hop_length)
    mag_original, phase_original = librosa.magphase(stft_original)
    
    # 4. 변형된 오디오의 STFT
    stft_modified = librosa.stft(modified, n_fft=n_fft, hop_length=hop_length)
    mag_modified, phase_modified = librosa.magphase(stft_modified)
    
    # 5. 원본 포먼트를 유지하면서 위상 변경 적용
    mag_result = mag_original
    phase_result = phase_modified
    
    # 6. ISTFT를 통해 최종 오디오 생성
    result = librosa.istft(mag_result * phase_result, hop_length=hop_length)
    
    # # 7. 랜덤 딜레이 추가 (5-15ms 사이)
    # delay_ms = np.random.uniform(5, 15)  # 5-15ms 사이의 랜덤 값
    # delay_samples = int(delay_ms * sr / 1000)  # ms를 샘플 수로 변환
    # result_delayed = np.pad(result, (delay_samples, 0), mode='constant')[:-delay_samples]
    
    # 8. 결과 저장
    sf.write(output_path, result, sr)
    
    #print(f"새로운 오디오 파일이 생성되었습니다: {output_path} (딜레이: {delay_ms:.1f}ms)")


def pitch_shift_with_formant_preservation(y, sr, semitones):
    """포먼트를 보존하면서 피치 시프트를 수행하는 함수"""
    # WORLD 분석
    _f0, t = pw.dio(y.astype(np.float64), sr)
    f0 = pw.stonemask(y.astype(np.float64), _f0, t, sr)
    sp = pw.cheaptrick(y.astype(np.float64), f0, t, sr)
    ap = pw.d4c(y.astype(np.float64), f0, t, sr)
    
    # 피치 변경
    modified_f0 = f0 * 2**(semitones/12)
    
    # 합성
    y_shifted = pw.synthesize(modified_f0, sp, ap, sr)
    return y_shifted


def test_formant_pitch_shift():
    """
    포먼트 보존 피치 시프트 테스트 함수
    """
    # 테스트할 입력 파일 경로
    input_path = "./dataset/song_1/Male Pop Rock Vocal C12.wav"  # 본인의 테스트 파일 경로로 변경
    output_dir = "./output_wavs/formant_test"
    os.makedirs(output_dir, exist_ok=True)
    
    # 오디오 로드
    y, sr = librosa.load(input_path, sr=None)
    
    # 다양한 피치 시프트 값으로 테스트
    semitones_list = [-12, -7, -5, -3, 0, 3, 5, 7, 12]  # 1옥타브 아래부터 위까지
    
    try:
        # 원본 파일 복사
        original_output = os.path.join(output_dir, "original.wav")
        sf.write(original_output, y, sr)
        print(f"원본 파일 저장됨: {original_output}")
        
        # 각각의 피치 시프트 값으로 테스트
        for semitones in semitones_list:
            output_name = f"formant_preserved_shift_{'+' if semitones >= 0 else ''}{semitones}.wav"
            output_path = os.path.join(output_dir, output_name)
            
            # 포먼트 보존 피치 시프트 적용
            y_shifted = pitch_shift_with_formant_preservation(y, sr, semitones)
            
            # 결과 저장
            sf.write(output_path, y_shifted, sr)
            print(f"생성 완료: {output_name} (피치 시프트: {semitones})")
            
        print("\n모든 테스트 완료!")
        print(f"결과 파일 위치: {output_dir}")
        
    except Exception as e:
        print(f"에러 발생: {str(e)}")


if __name__ == "__main__":
    test_formant_pitch_shift()
    # input_path = "./dataset/song_1/Male Pop Rock Vocal C12.wav"
    # output_dir = "./output_wavs/chord_parts"
    
    # # 메이저 코드 진행에 필요한 주요 화음들 생성
    # chords = ['I', 'ii', 'iii', 'IV', 'V', 'V7', 'vi']
    
    # try:
    #     for chord in chords:
    #         print(f"\n{chord} 화음 생성 중...")
    #         pitch_shift_chord(input_path, output_dir, scale_type='major', chord_name=chord)
            
    #     print("\n모든 화음 생성 완료!")
        
    # except Exception as e:
    #     print(f"에러 발생: {str(e)}")


