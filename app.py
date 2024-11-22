import streamlit as st
import os
from pitch_shift import (
    make_doubling, 
    pitch_shift_chord, 
    pitch_shift_with_formant_preservation,
    get_chord_intervals
)
import soundfile as sf
import zipfile
import io
import librosa
import uuid
import subprocess

def create_zip_file(files_dict):
    """
    파일들을 ZIP 파일로 압축
    files_dict: {'파일명': '파일경로'} 형태의 딕셔너리
    """
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for file_name, file_path in files_dict.items():
            zip_file.write(file_path, file_name)
    return zip_buffer

def main():
    st.title("AI 보컬 더블링 & 코러스 생성기")
    
    # 세션 상태 초기화
    if 'doubling_results' not in st.session_state:
        st.session_state['doubling_results'] = None
    if 'chorus_results' not in st.session_state:
        st.session_state['chorus_results'] = None
    if 'formant_chorus_results' not in st.session_state:
        st.session_state['formant_chorus_results'] = None
    
    # 프로젝트 소개
    st.markdown("""
    ### 🎤 녹음은 한 번만, 나머지는 AI가 해결합니다
    
    더 이상 같은 멜로디를 여러 번 부르지 마세요. AI가 자연스러운 더블링과 코러스를 만들어드립니다.
    
    #### ✨ 주요 기능
    - **원클릭 더블링**: 여러 번 녹음할 필요 없이, 한 번의 녹음으로 자연스러운 더블 트랙 생성
    - **AI 코러스**: 힘들게 음을 높이거나 낮춰 부를 필요 없이, 다양한 음역대의 코러스를 자동으로 생성
    - **자연스러운 결과**: AI와 최신 시그널 프로세싱으로 실제 녹음한 것 같은 미세한 차이 구현
    - **완벽한 제어**: 모든 트랙을 개별 WAV 파일로 제공하여 DAW에서 자유롭게 편집 가능
    
    #### 💡 이런 분들에게 추천합니다
    - 더블 트랙 녹음에 시간을 쓰기 어려운 뮤지션
    - 다양한 코러스를 시도해보고 싶은 보컬리스트
    - 자신만의 코러스 사운드를 만들고 싶은 프로듀서
    - 녹음 시간과 비용을 절약하고 싶은 작업자
    """)

    # 파일 업로드 섹션
    st.markdown("### 🎵 보컬 파일 업로드")
    uploaded_file = st.file_uploader("WAV 파일을 업로드해주세요", type=['wav'])
    
    if uploaded_file:
        # 새 파일이 업로드되면 세션 상태 초기화
        current_file_name = uploaded_file.name
        if 'current_file' not in st.session_state or st.session_state.current_file != current_file_name:
            st.session_state.current_file = current_file_name
            st.session_state.chorus_results = None
            st.session_state.formant_chorus_results = None

        st.write("원본 파일:")
        st.audio(uploaded_file, format='audio/wav')
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("### 🎭 더블링 생성")
            st.markdown("자연스러운 미세한 차이를 가진 더블 트랙을 생성합니다.")
            if st.button("더블링 생성", key="doubling"):
                try:
                    output_dir = "./output_wavs/doubles"
                    os.makedirs(output_dir, exist_ok=True)
                    
                    # 업로드된 파일 저장
                    original_filename = uploaded_file.name
                    temp_input_path = os.path.join(output_dir, original_filename)
                    with open(temp_input_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    with st.spinner("더블링 생성 중..."):
                        # 두 개의 더블링 생성
                        double1_path = os.path.join(output_dir, f"double1_{original_filename}")
                        double2_path = os.path.join(output_dir, f"double2_{original_filename}")
                        
                        make_doubling(temp_input_path, double1_path)
                        make_doubling(temp_input_path, double2_path)
                        
                        # 원본과 더블링들을 합치기
                        original, sr = librosa.load(temp_input_path, sr=None)
                        double1, _ = librosa.load(double1_path, sr=sr)
                        double2, _ = librosa.load(double2_path, sr=sr)
                        
                        # 모든 오디오를 가장 짧은 길이에 맞추기
                        min_length = min(len(original), len(double1), len(double2))
                        original = original[:min_length]
                        double1 = double1[:min_length]
                        double2 = double2[:min_length]
                        
                        # 볼륨 조절 (더블링은 원본보다 약간 작게)
                        mixed = original + 0.6 * double1 + 0.6 * double2
                        
                        # 클리핑 방지를 위한 정규화
                        mixed = librosa.util.normalize(mixed)
                        
                        # 합친 파일 저장
                        mixed_path = os.path.join(output_dir, f"mixed_{original_filename}")
                        sf.write(mixed_path, mixed, sr)
                    
                    st.success("더블링 생성 완료!")
                    
                    # 각각의 파일 재생
                    st.write("원본:")
                    st.audio(temp_input_path, format='audio/wav')
                    
                    st.write("더블링 1:")
                    st.audio(double1_path, format='audio/wav')
                    
                    st.write("더블링 2:")
                    st.audio(double2_path, format='audio/wav')
                    
                    st.write("모든 트랙 믹스:")
                    st.audio(mixed_path, format='audio/wav')
                    
                    # ZIP 다운로드 버튼 추가
                    files_dict = {
                        original_filename: temp_input_path,
                        f"double1_{original_filename}": double1_path,
                        f"double2_{original_filename}": double2_path,
                        f"mixed_{original_filename}": mixed_path
                    }
                    
                    zip_buffer = create_zip_file(files_dict)
                    st.download_button(
                        label="모든 파일 다운로드 (ZIP)",
                        data=zip_buffer.getvalue(),
                        file_name="vocal_doubles.zip",
                        mime="application/zip"
                    )
                    
                except Exception as e:
                    st.error(f"에러 발생: {str(e)}")
                st.session_state.doubling_results = {
                    'original': temp_input_path,
                    'double1': double1_path,
                    'double2': double2_path,
                    'mixed': mixed_path
                }
            
            # 더블링 결과 표시
            if st.session_state.doubling_results:
                st.write("더블링 결과:")
                for file_name, file_path in st.session_state.doubling_results.items():
                    st.write(f"파일명: {file_name}")
                    st.audio(file_path, format='audio/wav')
        
        with col2:
            st.markdown("### 🎹 코러스 생성")
            st.markdown("AI가 자연스러운 코러스를 만들어냅니다.")
            
            # 스케일과 코드 선택
            scale_col, chord_col = st.columns(2)
            with scale_col:
                scale_type = st.selectbox("스케일 선택", ['major', 'minor'])
            
            with chord_col:
                chord_options = ['I', 'ii', 'iii', 'IV', 'V', 'V7', 'vi'] if scale_type == 'major' else ['i', 'ii', 'III', 'iv', 'v', 'VI', 'VII', 'V7']
                chord_name = st.selectbox("코드 선택", chord_options)
            
            # 코러스 생성 1
            st.markdown("#### 코러스 생성 1")
            st.markdown("*음 변환*")
            if st.button(f"{chord_name} 코러스 생성", key="chorus_btn1"):
                try:
                    # 코러스 1용 디렉토리
                    output_dir = "./output_wavs/chorus1_parts"
                    os.makedirs(output_dir, exist_ok=True)
                    
                    # 기존 파일들 모두 삭제
                    for f in os.listdir(output_dir):
                        if f.endswith('.wav'):
                            os.remove(os.path.join(output_dir, f))
                    
                    # 업로드된 파일을 임시 저장
                    original_filename = uploaded_file.name
                    temp_input_path = os.path.join(output_dir, original_filename)
                    with open(temp_input_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    with st.spinner(f"{chord_name} 코러스를 생성하는 중..."):
                        # 코드 인터벌 가져오기
                        chord_intervals = get_chord_intervals(scale_type)
                        intervals = chord_intervals[chord_name]
                        
                        # 오디오 로드
                        y, sr = librosa.load(temp_input_path, sr=None)
                        
                        # 각 인터벌에 대해 포치 시프트 적용
                        files_dict = {original_filename: temp_input_path}  # 원본 파일 추가
                        shifted_signals = [y]  # 믹싱을 위한 신호 리스트 (원본 포함)
                        
                        for semitones in intervals:
                            # 각 인터벌마다 개별적으로 피치 시프트 적용
                            output_name = f"shifted_{scale_type}_{chord_name}_{'+' if semitones >= 0 else ''}{semitones}.wav"
                            output_path = os.path.join(output_dir, output_name)
                            
                            # soundstretch로 피치 시프트
                            cmd = ['soundstretch', temp_input_path, output_path, 
                                  f'-pitch={semitones}', '-tempo=0', '-rate=0', 
                                  '-speech', '-sequence=30', '-naa']
                            
                            subprocess.run(cmd, check=True)
                            
                            # 생성된 파일 로드 및 정규화
                            y_shifted, sr = librosa.load(output_path, sr=None)
                            y_shifted = librosa.util.normalize(y_shifted)
                            
                            # 결과 저장
                            sf.write(output_path, y_shifted, sr)
                            
                            shifted_signals.append(y_shifted)
                            files_dict[output_name] = output_path
                        
                        # 모든 신호를 가장 짧은 길이에 맞추기
                        min_length = min(len(signal) for signal in shifted_signals)
                        shifted_signals = [signal[:min_length] for signal in shifted_signals]
                        
                        # 믹싱 (원본 100% + 각 코러스 20%)
                        mixed = shifted_signals[0]  # 원본
                        for signal in shifted_signals[1:]:
                            mixed += 0.2 * signal
                        
                        # 클리핑 방지를 위한 정규화
                        mixed = librosa.util.normalize(mixed)
                        
                        # 믹스 파일 저장
                        mixed_path = os.path.join(output_dir, f"mixed_{original_filename}")
                        sf.write(mixed_path, mixed, sr)
                        files_dict['mixed'] = mixed_path
                        
                        st.session_state.chorus_results = files_dict
                        st.success(f"{chord_name} 코러스 생성 완료!")
                    
                except Exception as e:
                    st.error(f"에러 발생: {str(e)}")
            
            # 코러스 1 결과 표시
            if st.session_state.chorus_results:
                st.write("생성된 코러스 파일들:")
                for file_name, file_path in st.session_state.chorus_results.items():
                    if file_name != 'mixed':  # 개별 파일들 먼저 표시
                        st.write(f"파일명: {file_name}")
                        st.audio(file_path, format='audio/wav')
                
                # 믹스 파일 표시
                if 'mixed' in st.session_state.chorus_results:
                    st.write("모든 트랙 믹스:")
                    st.audio(st.session_state.chorus_results['mixed'], format='audio/wav')
            
            st.markdown("---")
            
            # 코러스 생성 2
            st.markdown("#### 코러스 생성 2")
            st.markdown("*포먼트 보존 기법 사용*")
            if st.button(f"{chord_name} 코러스 생성 (포먼트 보존)", key="chorus_btn2"):
                try:
                    # 코러스 2용 디렉토리
                    output_dir = "./output_wavs/chorus2_parts"
                    os.makedirs(output_dir, exist_ok=True)
                    
                    # 기존 파일들 모두 삭제
                    for f in os.listdir(output_dir):
                        if f.endswith('.wav'):
                            os.remove(os.path.join(output_dir, f))
                    
                    # 업로드된 파일을 임시 저장
                    original_filename = uploaded_file.name
                    temp_input_path = os.path.join(output_dir, original_filename)
                    with open(temp_input_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    with st.spinner(f"{chord_name} 코러스를 생성하는 중..."):
                        # 코드 인터벌 가져오기
                        chord_intervals = get_chord_intervals(scale_type)
                        intervals = chord_intervals[chord_name]
                        
                        # 오디오 로드
                        y, sr = librosa.load(temp_input_path, sr=None)
                        
                        # 각 인터벌에 대해 포먼트 보존 피치 시프트 적용
                        files_dict = {original_filename: temp_input_path}  # 원본 파일 추가
                        shifted_signals = [y]  # 믹싱을 위한 신호 리스트 (원본 포함)
                        
                        for semitones in intervals:
                            output_name = f"formant_preserved_{scale_type}_{chord_name}_{'+' if semitones >= 0 else ''}{semitones}.wav"
                            output_path = os.path.join(output_dir, output_name)
                            
                            # 포먼트 보존 피치 시프트 적용
                            y_shifted = pitch_shift_with_formant_preservation(y, sr, semitones)
                            
                            # 결과 저장
                            sf.write(output_path, y_shifted, sr)
                            files_dict[output_name] = output_path
                            shifted_signals.append(y_shifted)
                        
                        # 모든 신호를 가장 짧은 길이에 맞추기
                        min_length = min(len(signal) for signal in shifted_signals)
                        shifted_signals = [signal[:min_length] for signal in shifted_signals]
                        
                        # 믹싱 (원본 100% + 각 코러스 20%)
                        mixed = shifted_signals[0]  # 원본
                        for signal in shifted_signals[1:]:
                            mixed += 0.2 * signal
                        
                        # 클리핑 방지를 위한 정규화
                        mixed = librosa.util.normalize(mixed)
                        
                        # 믹스 파일 저장
                        mixed_path = os.path.join(output_dir, f"mixed_{original_filename}")
                        sf.write(mixed_path, mixed, sr)
                        files_dict['mixed'] = mixed_path
                        
                        st.session_state.formant_chorus_results = files_dict
                        st.success(f"{chord_name} 코러스 생성 완료!")
                    
                except Exception as e:
                    st.error(f"에러 발생: {str(e)}")
            
            # 코러스 2 결과 표시
            if st.session_state.formant_chorus_results:
                st.write("생성된 코러스 파일들 (포먼트 보존):")
                for file_name, file_path in st.session_state.formant_chorus_results.items():
                    if file_name != 'mixed':  # 개별 파일들 먼저 표시
                        st.write(f"파일명: {file_name}")
                        st.audio(file_path, format='audio/wav')
                
                # 믹스 파일 표시
                if 'mixed' in st.session_state.formant_chorus_results:
                    st.write("모든 트랙 믹스:")
                    st.audio(st.session_state.formant_chorus_results['mixed'], format='audio/wav')
    
    else:
        st.info("👆 먼저 WAV 파일을 업로드해주세요. 최상의 결과를 위해 깨끗한 녹음 파일을 사용하는 것을 추천드립니다.")
    


if __name__ == "__main__":
    main() 