import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
import soundfile as sf
from gtts import gTTS
import os
import librosa
from openai import OpenAI
from difflib import SequenceMatcher
import tempfile
import threading
import queue
import sys
import time

plt.rcParams['font.family'] = 'Malgun Gothic'  # 한글 폰트 설정

class EnglishPronunciationApp:
    @staticmethod
    def get_instance():
        if 'app_instance' not in st.session_state:
            st.session_state.app_instance = EnglishPronunciationApp()
        return st.session_state.app_instance

    def __init__(self):
        self.client = None
        self.recognized_text = ""
        self.original_text = ""
        if 'audio_data' not in st.session_state:
            st.session_state.audio_data = None
        if 'is_recording' not in st.session_state:
            st.session_state.is_recording = False
        if 'original_waveform' not in st.session_state:
            st.session_state.original_waveform = None

    def set_api_key(self, api_key):
        if api_key:
            self.client = OpenAI(api_key=api_key)
            return True
        return False

    def generate_speech(self, text):
        if text:
            self.original_text = text
            tts = gTTS(text=text, lang='en')
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio_file:
                tts.save(temp_audio_file.name)
                return temp_audio_file.name
        return None

    def start_recording(self):
        st.session_state.is_recording = True
        st.session_state.audio_data = None

    def stop_recording(self):
        st.session_state.is_recording = False

    def record_audio(self, duration=10):
        fs = 44100  # 샘플링 레이트
        recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
        
        # 카운트다운 표시
        countdown_placeholder = st.empty()
        for i in range(duration, 0, -1):
            countdown_placeholder.text(f"녹음 중... {i}초 남음")
            time.sleep(1)
        
        sd.wait()
        countdown_placeholder.text("녹음 완료!")

        return recording.flatten(), fs

    def transcribe_audio(self, audio_file):
        try:
            transcript = self.client.audio.transcriptions.create(
                model="whisper-1", 
                file=audio_file
            )
            return transcript.text
        except Exception as e:
            st.error(f"음성 인식 중 오류 발생: {str(e)}")
            return ""

    def calculate_similarity(self, text1, text2):
        return SequenceMatcher(None, text1.lower(), text2.lower()).ratio() * 100

    def analyze_pronunciation(self, original_text, recognized_text, similarity):
        prompt = f"""
        원본 텍스트: {original_text}
        인식된 텍스트: {recognized_text}
        텍스트 유사도: {similarity:.2f}%

        위의 두 텍스트를 비교하여 발음의 정확성을 분석해주세요. 다음 사항들을 고려해주세요:
        1. 단어의 누락 또는 추가
        2. 발음의 차이
        3. 강세와 억양의 문제
        4. 전반적인 유창성

        분석 결과를 한국어로 작성해주시고, 개선을 위한 구체적인 조언도 함께 제공해주세요.
        또한, 발음 점수를 100점 만점으로 매겨주세요.
        """

        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that analyzes English pronunciation."},
                    {"role": "user", "content": prompt}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            st.error(f"발음 분석 중 오류 발생: {str(e)}")
            return "발음 분석을 수행할 수 없습니다."

def plot_waveform(audio_data, sr, title):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(np.arange(len(audio_data)) / sr, audio_data)
    ax.set_xlabel('시간 (초)')
    ax.set_ylabel('진폭')
    ax.set_title(title)
    plt.tight_layout()
    return fig

def main():
    st.set_page_config(page_title="영어 발음 분석기", layout="wide")
    st.title("영어 발음 분석기")

    app = EnglishPronunciationApp.get_instance()

    # API Key 입력
    if 'api_key' not in st.session_state:
        st.session_state.api_key = ''

    api_key = st.text_input("OpenAI API Key:", type="password", value=st.session_state.api_key)
    if st.button("API 키 설정"):
        if app.set_api_key(api_key):
            st.session_state.api_key = api_key
            st.success("API 키가 설정되었습니다.")
        else:
            st.error("API 키를 입력해주세요.")

    # 텍스트 입력
    text_input = st.text_input("영어 문장 입력:")
    if st.button("음성 생성"):
        if not app.client:
            st.error("먼저 API 키를 설정해주세요.")
        else:
            audio_file = app.generate_speech(text_input)
            if audio_file:
                st.audio(audio_file, format="audio/mp3")
                y, sr = librosa.load(audio_file)
                st.session_state.original_waveform = plot_waveform(y, sr, "원본 음성")
                st.pyplot(st.session_state.original_waveform)
                os.unlink(audio_file)  # 임시 파일 삭제

    # 원본 파형 표시 (지속적으로)
    if st.session_state.original_waveform:
        st.pyplot(st.session_state.original_waveform)

    # 녹음 버튼
    if st.button("녹음 시작"):
        if not app.client:
            st.error("먼저 API 키를 설정해주세요.")
        else:
            st.warning("녹음을 할 때 녹음은 10초 동안 진행됩니다.")
            app.start_recording()
            with st.spinner("녹음 중..."):
                recording, fs = app.record_audio()
            st.success("녹음 완료!")
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio_file:
                sf.write(temp_audio_file.name, recording, fs)
                st.audio(temp_audio_file.name, format="audio/wav")
                st.pyplot(plot_waveform(recording, fs, "녹음된 음성"))

                with open(temp_audio_file.name, "rb") as audio_file:
                    recognized_text = app.transcribe_audio(audio_file)

            st.write(f"인식된 텍스트: {recognized_text}")

            if app.original_text:
                similarity = app.calculate_similarity(app.original_text, recognized_text)
                analysis = app.analyze_pronunciation(app.original_text, recognized_text, similarity)
                
                st.subheader("분석 결과")
                st.write(f"텍스트 유사도: {similarity:.2f}%")
                st.write(analysis)

            os.unlink(temp_audio_file.name)  # 임시 파일 삭제

if __name__ == "__main__":
    main()