import librosa
import os
import numpy as np
import scipy.io.wavfile as wavfile
import torch
import soundfile as sf
import torch.nn as nn
import pyworld as pw
import pyrubberband as pyrb
import subprocess
import tempfile
import time



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

import os
import librosa
import soundfile as sf
import numpy as np
import pyworld as pw
import torch

def pitch_shift_with_formants(result_delayed, sr, semitones):
    """포먼트를 보존하면서 피치 시프팅"""

    if semitones == 0:
        return result_delayed

    # 데이터를 64비트 float로 변환
    result_delayed = result_delayed.astype(np.float64)

    # World Vocoder를 이용해 음성을 분석
    _f0, timeaxis = pw.harvest(result_delayed, sr)  # 피치 추출
    sp = pw.cheaptrick(result_delayed, _f0, timeaxis, sr)  # 스펙트럼 추출
    ap = pw.d4c(result_delayed, _f0, timeaxis, sr)  # 에어리니스 추출

    # 피치 변경
    f0_shifted = _f0 * (2 ** (semitones / 12.0))  # 세미톤 단위로 피치 변경

    # World Vocoder를 사용하여 재합성
    y_shifted = pw.synthesize(f0_shifted, sp, ap, sr)
    return y_shifted

def pitch_shift_chord(input_path, output_path, scale_type='major', chord_name='I'):
    # output_dir을 output_path의 디렉토리로 설정
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    
    # 생성된 파일들의 경로를 저장할 리스트
    generated_files = []
    
    # chord_intervals 가져오기
    chord_intervals = get_chord_intervals(scale_type)
    
    try:
        for semitones in chord_intervals[chord_name]:
            # 각 음정마다 새로운 더블링 생성
            random_seed = np.random.randint(0, 10000)
            torch.manual_seed(random_seed)
            
            # 임시 더블링 파일 경로
            doubled_path = os.path.join(output_dir, f"doubled_temp_{semitones}.wav")
            
            # make_doubling 함수를 사용하여 더블링 생성
            make_doubling(input_path, doubled_path)
            
            # 더블링된 파일 로드
            result_delayed, sr = librosa.load(doubled_path, sr=None)

            # max_val = 0.8
            # result_delayed = np.clip(result_delayed, -max_val, max_val)

            # # 스무딩 필터 적용
            # window_size = 5  # 윈도우 크기 설정
            # result_delayed = np.convolve(result_delayed, np.ones(window_size)/window_size, mode='same')

            # soundstretch 대신 World Vocoder로 피치 시프트
            base_filename = os.path.splitext(os.path.basename(input_path))[0]
            output_name = f"{base_filename}_doubled_{scale_type}_{chord_name}_{'+' if semitones >= 0 else ''}{semitones}.wav"
            output_path = os.path.join(output_dir, output_name)
            
            try:
                # World Vocoder 기반 피치 시프트
                y_shifted = pitch_shift_with_formants(result_delayed, sr, semitones)
                
                # 정규화
                y_shifted = librosa.util.normalize(y_shifted)
                
                # 결과 저장
                sf.write(output_path, y_shifted, sr)
                print(f'생성 완료: {output_name} (피치 시프트: {semitones})')
                
                # 생성된 파일 경로 가
                generated_files.append(output_path)
                
            except Exception as e:
                print(f"개별 파일 처리 실패: {str(e)}")
                if os.path.exists(output_path):
                    os.remove(output_path)
                continue
            
            # 임시 파일 삭제
            if os.path.exists(doubled_path):
                os.remove(doubled_path)
        
        return generated_files  # 생성된 파일 목록 반환
        
    except Exception as e:
        print(f"전체 처리 실패: {str(e)}")
        return []




def make_doubling(input_path, output_path, seed=None):
    """
    입력된 WAV 파일의 더블링을 생성하는 함수
    
    Args:
        input_path: 입력 WAV 파일 경로
        output_path: 출력 WAV 파일 경로
        seed: 랜덤 시드 (None이면 현재 시간 기반으로 생성)
    """
    # 랜덤 시드 설정
    if seed is None:
        seed = int(time.time() * 1000) % 10000
    torch.manual_seed(seed)
    np.random.seed(seed)
    
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
    
    # 랜덤한 미세 딜레이 생성 (5-15ms 사이)
    delay_ms = np.random.uniform(5, 15)
    delay_samples = int((delay_ms / 1000) * sr)
    
    # 딜레이된 신호 생성
    result_delayed = np.zeros(len(result) + delay_samples)
    result_delayed[delay_samples:] = result
    
    # 8. 결과 저장
    sf.write(output_path, result_delayed, sr)
    
    #print(f"새로운 오디오 파일이 생성되었습니다: {output_path} (딜레이: {delay_ms:.1f}ms)")


def pitch_shift_with_formant_preservation(y, sr, semitones):
    """포먼트를 보존하면서 피치 시프트를 수행하는 함수"""
    try:
        # 임시 파일 생성을 위한 경로 설정
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_input, \
             tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_doubled:
            
            # 입력 신호를 임시 파일로 저장
            sf.write(temp_input.name, y, sr)
            
            # 더블링 생성
            make_doubling(temp_input.name, temp_doubled.name)
            
            # 더블링된 파일 로드
            y_doubled, sr = librosa.load(temp_doubled.name, sr=None)
            
            # 피치 시프트 범위 제한
            semitones = np.clip(semitones, -12, 12)
            
            try:
                # pyrubberband를 사용한 피치 시프트
                y_shifted = pyrb.pitch_shift(y_doubled, sr, n_steps=semitones)
            except Exception as e:
                print(f"pyrubberband 처리 중 오류, librosa로 대체: {str(e)}")
                y_shifted = librosa.effects.pitch_shift(y=y_doubled, sr=sr, n_steps=semitones)
            
            # 볼륨 정규화
            y_shifted = librosa.util.normalize(y_shifted)
            
            # 임시 파일 삭제
            os.unlink(temp_input.name)
            os.unlink(temp_doubled.name)
            
            return y_shifted
            
    except Exception as e:
        print(f"피치 시프트 처리 중 오류 발생: {str(e)}")
        # 에러 발생 시 원본 신호에 대해 librosa의 pitch_shift 적용
        y_shifted = librosa.effects.pitch_shift(y=y, sr=sr, n_steps=semitones)
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
    #test_formant_pitch_shift()
    input_path = "./dataset/song_1/Male Pop Rock Vocal C12.wav"
    output_dir = "./output_wavs/chord_parts5"
    
    # 메이저 코드 진행에 필요한 주요 화음들 생성
    chords = ['I', 'ii', 'iii', 'IV', 'V', 'V7', 'vi']
    

    pitch_shift_chord(input_path, output_dir, scale_type='major', chord_name='vi')
            


