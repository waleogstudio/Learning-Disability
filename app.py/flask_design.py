from flask import Flask, render_template, request, session
from main_design import eye_preprocess, hand_write, eeg_preprocess,splitdata ,transformer_model ,build_BiLSTM_model ,create_and_train_model,hirarcial_cross_attention,final_model_process                                               
from main_design import model_metrics
import numpy as np
import os
import matplotlib.pyplot as plt
import io
import base64
app = Flask(__name__)
app.secret_key = 'supersecretkey'  # Required for session to work

@app.route('/')
def index():
    return render_template('index.html')



# Global variables to hold features
eye_x, eye_y, eeg_x, handwrite_x = None, None, None, None


@app.route('/process', methods=['GET', 'POST'])
def process_page():
    global eye_x, eye_y, eeg_x, handwrite_x, eeg_correct_signals, eeg_non_correct_signals
    global eeg_plot_signals  # Used to pass signals to /eeg_plot_route

    if request.method == 'GET':
        session.pop('data_eye', None)
        session.pop('samples_eye', None)
        session.pop('data_hw', None)
        session.pop('data_eeg', None)
        session.pop('samples_eeg', None)

    data_eye = session.get('data_eye')
    samples_eye = session.get('samples_eye')
    data_hw = session.get('data_hw')
    data_eeg = session.get('data_eeg')
    samples_eeg = session.get('samples_eeg')

    if request.method == 'POST':
        if 'eye' in request.form:
            dyslexia, non_dyslexia, eye_x, eye_y = eye_preprocess( r'C:\Users\waleo\OneDrive\Desktop\Learning_Disability_dataset\eye_tracking\data')
            data_eye = f"✅ Eye preprocessing completed! Dyslexia: {len(dyslexia)} samples, Non-Dyslexia: {len(non_dyslexia)} samples"
            samples_eye = dyslexia[:5].tolist()
            session['data_eye'] = data_eye
            session['samples_eye'] = samples_eye

        if 'handwrite' in request.form:
            normal, reversal, handwrite_x = hand_write(r"C:\Users\waleo\OneDrive\Desktop\Learning_Disability_dataset\Handwriting_dataset\Gambo")
            data_hw = f"✅ Handwriting preprocessing completed! Normal: {len(normal)} samples, Reversal: {len(reversal)} samples"
            session['data_hw'] = data_hw
            
        if 'eeg' in request.form:
            correct, non_correct, total_correct_ica, total_non_correct_ica, total_correct_bwf, total_non_correct_bwf, eeg_x = eeg_preprocess(r"C:\Users\waleo\OneDrive\Desktop\Learning_Disability_dataset\EEG-dataset\ds002680")
            eeg_correct_signals = correct
            eeg_non_correct_signals = non_correct
            data_eeg = f"✅ EEG preprocessing completed! Correct: {len(correct)} samples, Non-Correct: {len(non_correct)} samples"
            samples_eeg = correct[:5].tolist()
            session['data_eeg'] = data_eeg
            session['samples_eeg'] = samples_eeg

            # Select first valid pair (non-identical)
            correct_signal = None
            non_correct_signal = None
            for c in correct:
                for nc in non_correct:
                    if not np.array_equal(np.array(c), np.array(nc)):
                        correct_signal = c
                        non_correct_signal = nc
                        break
                if correct_signal is not None:
                    break

            if correct_signal is not None and non_correct_signal is not None:
                eeg_plot_signals = (
                    correct_signal,
                    non_correct_signal,
                    total_correct_ica[0],
                    total_non_correct_ica[0],
                    total_correct_bwf[0],
                    total_non_correct_bwf[0]
                )
            else:
                eeg_plot_signals = None

    return render_template(
        'code_process.html',
        data_eye=data_eye,
        samples_eye=samples_eye,
        data_hw=data_hw,
        data_eeg=data_eeg,
        samples_eeg=samples_eeg
    )


def clear_old_plots():
    image_dir = os.path.join(app.root_path, 'static', 'images')
    for f in os.listdir(image_dir):
        if f.startswith("eeg_plot_") and f.endswith(".png"):
            os.remove(os.path.join(image_dir, f))


@app.route('/eeg_plot_route')
def eeg_plot_route():
    global eeg_plot_signals

    if eeg_plot_signals is None:
        return render_template("eeg_plot.html", plot_files=[], message="⚠️ No distinct EEG signals available. Please preprocess EEG first.")

    clear_old_plots()

    def save_plot(signal, title, index, x_range=None):
        signal = np.array(signal).squeeze()

        if signal.ndim != 1 or len(signal) == 0:
            print(f"Warning: Signal {index} is invalid or empty.")
            return None

        plt.figure(figsize=(10, 4))

        if x_range and len(signal) > x_range[1]:
            # Plot only the selected range
            plt.plot(range(x_range[0], x_range[1]), signal[x_range[0]:x_range[1]])
        else:
            # Plot full signal
            plt.plot(signal)

        plt.title(title)
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        filename = f"eeg_plot_{index}.png"
        filepath = os.path.join(app.root_path, 'static', 'images', filename)
        plt.savefig(filepath)
        plt.close()
        return filename

    titles = [
        "Correct EEG Signal",
        "Non-Correct EEG Signal",
        "Correct EEG (ICA)",
        "Non-Correct EEG (ICA)",
        "Correct EEG (Bandpass)",
        "Non-Correct EEG (Bandpass)"
    ]

    plot_filenames = []
    for idx, (sig, title) in enumerate(zip(eeg_plot_signals, titles)):
        if idx == 1:
            plot_file = save_plot(sig, title, idx, x_range=(200, 800))
        else:
            plot_file = save_plot(sig, title, idx)

        if plot_file:
            plot_filenames.append(plot_file)

    return render_template("eeg_plot.html", plot_files=plot_filenames, message="✅ 6 EEG Plots Generated")





# Global variables
eye_x, eye_y, eeg_x, handwrite_x = None, None, None, None
eye_x_train, eye_x_test, eeg_x_train, eeg_x_test, hw_x_train, hw_x_test, y_train, y_test = None, None, None, None, None, None, None, None

@app.route('/model_execute', methods=['GET', 'POST'])
def model():
    global eye_x, eye_y, eeg_x, handwrite_x
    global eye_x_train, eye_x_test, eeg_x_train, eeg_x_test, hw_x_train, hw_x_test, y_train, y_test
    global eye_train_features, eeg_train_features, hw_train_features
    global eye_test_features, eeg_test_features, hw_test_features  # Ensure these are included

    if eye_x is None or eeg_x is None or handwrite_x is None or eye_y is None:
        return render_template('model_execute.html', message="❌ Missing data. Please preprocess first.")

    message = "Click the buttons to run your models."

    # ✅ Data Splitting
    if request.method == 'POST' and 'split' in request.form:
        eye_x_train, eye_x_test, eeg_x_train, eeg_x_test, hw_x_train, hw_x_test, y_train, y_test = splitdata(
            eye_x, eeg_x, handwrite_x, eye_y
        )

        sample_msg = (
            "✅ Data Split Done!\n"
            f"First 5 Eye Training Samples:\n{eye_x_train}\n\n"
            "🔍 Data Shapes:\n"
            f"Eye X Train: {eye_x_train.shape}, Eye X Test: {eye_x_test.shape}\n"
            f"EEG X Train: {eeg_x_train.shape}, EEG X Test: {eeg_x_test.shape}\n"
            f"HW X Train: {hw_x_train.shape}, HW X Test: {hw_x_test.shape}\n"
            f"Y Train: {y_train.shape}, Y Test: {y_test.shape}"
        )
        return render_template('model_execute.html', message=sample_msg)

    # 🚀 Transformer for Eye Data
    if request.method == 'POST' and 'transformer' in request.form:
        try:
            eye_train_features, eye_test_features, t_model, aacc = transformer_model(
                eye_x_train, y_train, eye_x_test, y_test
            )
            message = f"✅ Transformer Model (Eye) Executed Successfully!\nAccuracy: {aacc:.2f}"
        except Exception as e:
            message = f"❌ Error in Transformer Model: {str(e)}"

    # 🔁 BiLSTM for EEG Data
    if request.method == 'POST' and 'bilstm' in request.form:
        try:
            eeg_train_features, eeg_test_features, acc = build_BiLSTM_model(
                eeg_x_train, y_train, eeg_x_test, y_test
            )
            message = f"✅ BiLSTM Model (EEG) Executed Successfully!\nAccuracy: {acc:.2f}"
        except Exception as e:
            message = f"❌ Error in BiLSTM Model: {str(e)}"

    # 🧠 ViT for Handwriting Data
    if request.method == 'POST' and 'vit' in request.form:
        try:
            hw_train_features, hw_test_features, vit_acc = create_and_train_model(
                hw_x_train, hw_x_test, y_train, y_test
            )
            message = f"✅ ViT Model (Handwriting) Executed Successfully!\nAccuracy: {vit_acc:.2f}"
        except Exception as e:
            message = f"❌ Error in ViT Model: {str(e)}"

    return render_template('model_execute.html', message=message)


model = None
x_test = None
y_test = None

@app.route('/final', methods=['GET', 'POST'])
def final_model():
    global eye_train_features, eeg_train_features, hw_train_features
    global eye_test_features, eeg_test_features, hw_test_features
    global y_train, y_test, processed_x_train, processed_x_test
    global model, x_test

    # Safely initialize in case globals were never set
    eye_train_features = eye_train_features if 'eye_train_features' in globals() else None
    eeg_train_features = eeg_train_features if 'eeg_train_features' in globals() else None
    hw_train_features = hw_train_features if 'hw_train_features' in globals() else None
    eye_test_features = eye_test_features if 'eye_test_features' in globals() else None
    eeg_test_features = eeg_test_features if 'eeg_test_features' in globals() else None
    hw_test_features = hw_test_features if 'hw_test_features' in globals() else None

    message = "Click the buttons to run processes"
    accuracy = None
    x_train_display = None
    x_test_shape = None
    hierarchical_clicked = False
    preprocessing_done = False

    if request.method == 'POST':
        if 'hierarchical' in request.form:
            try:
                if any(v is None for v in [eye_train_features, eeg_train_features, hw_train_features,
                                           eye_test_features, eeg_test_features, hw_test_features]):
                    raise ValueError("❌ Please run all three models (Transformer, BiLSTM, ViT) before applying Hierarchical Cross-Attention.")

                processed_x_train, processed_x_test, y_train, y_test = hirarcial_cross_attention(
                    eye_train_features, eeg_train_features, hw_train_features,
                    eye_test_features, eeg_test_features, hw_test_features, y_train, y_test
                )
                x_train_display = f"Processed X Train Shape: {processed_x_train.shape}"
                x_test_shape = f"Processed X Test Shape: {processed_x_test.shape}"
                message = "✅ Hierarchical Cross-Attention Applied Successfully!"
                hierarchical_clicked = True
                preprocessing_done = True
            except Exception as e:
                message = f"❌ Error in Hierarchical Cross-Attention: {str(e)}"

        elif 'final_model' in request.form:
            try:
                if 'processed_x_train' not in globals() or processed_x_train is None:
                    if any(v is None for v in [eye_train_features, eeg_train_features, hw_train_features,
                                               eye_test_features, eeg_test_features, hw_test_features]):
                        raise ValueError("❌ Please run all three models before final fusion.")
                    x_train = np.concatenate([eye_train_features, eeg_train_features, hw_train_features], axis=1)
                    x_test = np.concatenate([eye_test_features, eeg_test_features, hw_test_features], axis=1)
                    x_train_display = f"Using concatenated features. Shape: {x_train.shape}"
                    x_test_shape = f"Test features shape: {x_test.shape}"
                else:
                    x_train, x_test = processed_x_train, processed_x_test

                accuracy, model = final_model_process(x_train, x_test, y_train, y_test)
                message = f"✅ Final Model Executed Successfully!"
            except Exception as e:
                message = f"❌ Error in Final Model: {str(e)}"

    return render_template('final.html',
                           message=message,
                           accuracy=accuracy if accuracy is not None else 0.0,
                           x_train_display=x_train_display,
                           x_test_shape=x_test_shape,
                           hierarchical_clicked=hierarchical_clicked,
                           preprocessing_done=preprocessing_done)



@app.route('/metrices')
def metrics_page():
    global model, x_test, y_test

    if model is None or x_test is None or y_test is None:
        return render_template('metrices.html', message="❌ Model or data not available. Run final model first.")

    # Get raw metric values
    accuracy, precision, f1score, sensitivity, specificity, auc, y_predicted = model_metrics(model, x_test, y_test)

    metrics_names = ['Accuracy', 'Precision', 'F1 Score', 'Sensitivity', 'Specificity', 'AUC']
    values = [accuracy, precision, f1score, sensitivity, specificity, auc]
    colors = ['#007bff', '#28a745', '#ffc107', '#dc3545', '#6f42c1', '#17a2b8']

    # Plot with original values (0.xxxxxx) — no conversion to % or rounding
    plt.figure(figsize=(10, 5))
    bars = plt.bar(metrics_names, values, color=colors)
    plt.ylim(0, 1.05)
    plt.title('📊 Model Evaluation Metrics')
    plt.ylabel('Score')

    # Add 7-digit values on top of each bar
    for bar, val in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2.0, val + 0.01, f'{val:.7f}', ha='center', fontsize=10)

    # Save plot to buffer
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    plt.close()

    # Send original values to the HTML template
    return render_template('metrices.html',
                           accuracy=f"{accuracy:.7f}",
                           precision=f"{precision:.7f}",
                           f1score=f"{f1score:.7f}",
                           sensitivity=f"{sensitivity:.7f}",
                           specificity=f"{specificity:.7f}",
                           auc=f"{auc:.7f}",
                           plot_url=img_base64)


if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)




    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    