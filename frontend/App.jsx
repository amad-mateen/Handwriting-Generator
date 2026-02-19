const { useState } = React;

const App = () => {
    const [text, setText] = useState('');
    const [animate, setAnimate] = useState(false);
    const [result, setResult] = useState(null);
    const [error, setError] = useState(null);
    const [loading, setLoading] = useState(false);

    const handleTextChange = (e) => {
        const newText = e.target.value;
        setText(newText);
    };

    const handleGenerate = async () => {
        setLoading(true);
        setError(null);
        setResult(null);

        try {
            const response = await fetch('/generate', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ text, animate }),
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || 'Failed to generate handwriting');
            }

            const blob = await response.blob();
            const url = URL.createObjectURL(blob);
            setResult(url);
        } catch (err) {
            console.error(err);  // Log the full error to the console
            setError(err.message);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="container">
            <h1 className="title">Handwriting Generator</h1>
            
            <div className="mb-4">
                <label htmlFor="textInput" className="input-label">
                    Enter Text:
                </label>
                <input
                    id="textInput"
                    type="text"
                    value={text}
                    onChange={handleTextChange}
                    className="text-input"
                    placeholder="Type your text here..."
                />
            </div>

            <div className="checkbox-container">
                <input
                    id="animateCheckbox"
                    type="checkbox"
                    checked={animate}
                    onChange={(e) => setAnimate(e.target.checked)}
                    className="checkbox"
                />
                <label htmlFor="animateCheckbox" className="checkbox-label">
                    Animate the handwriting
                </label>
            </div>

            <button
                onClick={handleGenerate}
                disabled={loading || !text}
                className="generate-button"
            >
                {loading ? 'Generating...' : 'Generate'}
            </button>

            {error && (
                <div className="error-message">
                    Error: {error}
                </div>
            )}

            {result && (
                <div className="result-container">
                    <h2 className="result-title">Generated Handwriting:</h2>
                    <img
                        src={result}
                        alt="Generated Handwriting"
                        className="generated-image"
                    />
                </div>
            )}

            <footer className="footer">
                © 2025 Handwriting Synthesis App. All rights reserved.
            </footer>
        </div>
    );
};

// Render the app
const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(<App />);