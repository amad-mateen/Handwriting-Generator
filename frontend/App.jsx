const { useState, useEffect } = React;

const VOCAB_CHARS = " !\"#'()+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";

const App = () => {
    const [text, setText] = useState('');
    const [animate, setAnimate] = useState(false);
    const [bias, setBias] = useState(10.0);
    const [stylePreset, setStylePreset] = useState('');
    const [result, setResult] = useState(null);
    const [error, setError] = useState(null);
    const [loading, setLoading] = useState(false);
    const [invalidChars, setInvalidChars] = useState([]);

    // Validate text input on the fly
    useEffect(() => {
        const invalid = [];
        for (let char of text) {
            if (!VOCAB_CHARS.includes(char)) {
                if (!invalid.includes(char)) {
                    invalid.push(char);
                }
            }
        }
        setInvalidChars(invalid);
    }, [text]);

    const handleTextChange = (e) => {
        setText(e.target.value);
    };

    const handleGenerate = async () => {
        if (invalidChars.length > 0 || !text) return;

        setLoading(true);
        setError(null);
        setResult(null);

        try {
            const bodyData = { 
                text, 
                animate, 
                bias: parseFloat(bias) 
            };
            
            if (stylePreset !== '') {
                bodyData.style_preset = parseInt(stylePreset, 10);
            }

            const response = await fetch('/generate', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(bodyData),
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || 'Failed to generate handwriting');
            }

            const blob = await response.blob();
            const url = URL.createObjectURL(blob);
            setResult(url);
        } catch (err) {
            console.error("[UI Error]", err);
            setError(err.message);
        } finally {
            setLoading(false);
        }
    };

    // Clean up object URL when component unmounts or result changes
    useEffect(() => {
        return () => {
            if (result) {
                URL.revokeObjectURL(result);
            }
        };
    }, [result]);

    return (
        <div className="container p5-theme">
            <header className="header p5-header">
                <div className="title-banner">
                    <h1 className="title">HANDWRITING SYNTHESIS</h1>
                </div>
                <p className="subtitle">
                    LSTM MDN SEQUENCE GENERATOR / DEEP LEARNING MODEL
                </p>
            </header>
            
            <div className="form-group p5-input-group">
                <label htmlFor="textInput" className="input-label p5-label">
                    ENTER TARGET TEXT:
                </label>
                <input
                    id="textInput"
                    type="text"
                    value={text}
                    onChange={handleTextChange}
                    className={`text-input p5-input ${invalidChars.length > 0 ? 'input-error' : ''}`}
                    placeholder="TYPE HERE..."
                    maxLength={100}
                />
                
                {invalidChars.length > 0 && (
                    <div className="validation-warning p5-warning">
                        SYSTEM FAULT: INVALID CHARACTERS DETECTED - {invalidChars.map(c => `'${c}'`).join(', ')}
                    </div>
                )}
            </div>

            <div className="settings-row p5-settings">
                <div className="form-group mb-4">
                    <label htmlFor="styleSelect" className="input-label p5-label">
                        HANDWRITING STYLE (EXPERIMENTAL):
                    </label>
                    <select
                        id="styleSelect"
                        value={stylePreset}
                        onChange={(e) => setStylePreset(e.target.value)}
                        className="style-select p5-select"
                    >
                        <option value="">DEFAULT NETWORK STYLE</option>
                        <option value="0">STYLE A - SLANTED CURSIVE</option>
                        <option value="3">STYLE B - BOLD & LARGE</option>
                        <option value="5">STYLE C - COMPACT & NEAT</option>
                        <option value="12">STYLE D - LOOSE & FLOWING</option>
                        <option value="14">STYLE E - NEAT PRINT</option>
                    </select>
                </div>

                <div className="slider-group p5-slider-group">
                    <div className="slider-header">
                        <label htmlFor="biasSlider" className="slider-label p5-label">
                            SAMPLING BIAS: <span className="p5-bias-val">{bias.toFixed(1)}</span>
                        </label>
                        <span className="bias-desc p5-desc">
                            {bias < 2.0 ? "CREATIVE" : bias > 11.0 ? "UNIFORM" : "NATURAL"}
                        </span>
                    </div>
                    <input
                        id="biasSlider"
                        type="range"
                        min="0.1"
                        max="15.0"
                        step="0.1"
                        value={bias}
                        onChange={(e) => setBias(parseFloat(e.target.value))}
                        className="bias-slider p5-slider"
                    />
                </div>
            </div>

            <div className="checkbox-container p5-checkbox-container">
                <input
                    id="animateCheckbox"
                    type="checkbox"
                    checked={animate}
                    onChange={(e) => setAnimate(e.target.checked)}
                    className="checkbox p5-checkbox"
                />
                <label htmlFor="animateCheckbox" className="checkbox-label p5-checkbox-label">
                    ANIMATE PEN PATH (OUTPUT AS GIF)
                </label>
            </div>

            <button
                onClick={handleGenerate}
                disabled={loading || !text || invalidChars.length > 0}
                className="generate-button p5-btn"
            >
                {loading ? (
                    <span className="spinner-container">
                        <span className="spinner p5-spinner"></span>
                        SYNTHESIZING...
                    </span>
                ) : 'SYNTHESIZE STROKES'}
            </button>

            {error && (
                <div className="error-message p5-error">
                    <strong>EXECUTION CRASHED:</strong> {error}
                </div>
            )}

            {result && (
                <div className="result-container p5-result animate-fade-in">
                    <div className="result-header p5-result-header">
                        <h2 className="result-title p5-title">GENERATED ASSET:</h2>
                        <a
                            href={result}
                            download={`handwriting_${text.substring(0, 15).replace(/[^a-zA-Z0-9]/g, '_')}.${animate ? 'gif' : 'png'}`}
                            className="download-button p5-download-btn"
                            title="Download file"
                        >
                            GET FILE
                        </a>
                    </div>
                    <div className="image-wrapper p5-canvas">
                        <img
                            src={result}
                            alt="Generated Handwriting Strokes"
                            className="generated-image"
                        />
                    </div>
                </div>
            )}

            <footer className="footer p5-footer">
                <p>PYTORCH LSTM HANDWRITING GENERATOR IMPLEMENTED BY AMAD MATEEN</p>
                <p>THEORETICAL ARCHITECTURE INSPIRED BY ALEX GRAVES (2013)</p>
                <p>© 2026 HANDWRITING SYNTHESIS SYSTEM. SHOWCASE VERSION.</p>
            </footer>
        </div>
    );
};

// Mount App root
const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(<App />);