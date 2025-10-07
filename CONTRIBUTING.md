# Contributing Guide

Thank you for your interest in contributing to the Marine Organism Identification System!

## Development Setup

### Prerequisites
- Python 3.9+
- Node.js 18+
- Git

### Setup Steps

1. **Fork and Clone**
   ```bash
   git clone https://github.com/yourusername/Dashboard.git
   cd Dashboard
   ```

2. **Backend Setup**
   ```bash
   cd backend
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   pip install -e .  # Install in development mode
   ```

3. **Frontend Setup**
   ```bash
   cd frontend
   npm install
   ```

4. **Run Tests**
   ```bash
   # Backend tests
   cd backend
   pytest tests/ -v

   # Frontend tests (if added)
   cd frontend
   npm test
   ```

## Code Style

### Python (Backend)
- Follow PEP 8
- Use type hints
- Maximum line length: 100 characters
- Use docstrings for all functions/classes

```python
def predict_frame(self, image: np.ndarray) -> Dict:
    """
    Predict species from image frame.
    
    Args:
        image: Input image as numpy array (H, W, C)
        
    Returns:
        Dictionary with prediction results
    """
    pass
```

### JavaScript/React (Frontend)
- Use ES6+ features
- Functional components with hooks
- PropTypes or TypeScript for type checking
- 2-space indentation

```javascript
export const CameraCapture = ({ onFrameCapture, isCapturing }) => {
  // Component implementation
};
```

## Commit Guidelines

### Commit Message Format
```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting)
- `refactor`: Code refactoring
- `test`: Adding tests
- `chore`: Maintenance tasks

**Examples:**
```
feat(backend): add support for ONNX models

- Implement ONNX runtime inference
- Add model conversion script
- Update documentation

Closes #123
```

```
fix(frontend): resolve camera permission issue on Safari

- Add fallback for getUserMedia
- Improve error messaging
- Test on Safari 16+
```

## Pull Request Process

1. **Create Feature Branch**
   ```bash
   git checkout -b feat/your-feature-name
   ```

2. **Make Changes**
   - Write clean, documented code
   - Add tests for new features
   - Update documentation

3. **Test Thoroughly**
   ```bash
   # Run all tests
   pytest tests/
   npm test
   
   # Test manually
   npm run dev
   ```

4. **Commit Changes**
   ```bash
   git add .
   git commit -m "feat(scope): description"
   ```

5. **Push and Create PR**
   ```bash
   git push origin feat/your-feature-name
   ```
   - Create PR on GitHub
   - Fill out PR template
   - Link related issues

6. **Code Review**
   - Address review comments
   - Keep PR focused and small
   - Rebase if needed

## Areas for Contribution

### High Priority
- [ ] Additional model architectures (YOLO, Faster R-CNN)
- [ ] Mobile app (React Native)
- [ ] Advanced analytics dashboard
- [ ] Multi-language support
- [ ] Accessibility improvements

### Medium Priority
- [ ] Video recording with annotations
- [ ] CSV/Excel export functionality
- [ ] Cloud storage integration
- [ ] Alert system for specific species
- [ ] Performance optimizations

### Good First Issues
- [ ] Improve error messages
- [ ] Add more unit tests
- [ ] Documentation improvements
- [ ] UI/UX enhancements
- [ ] Bug fixes

## Testing Guidelines

### Backend Tests
```python
# tests/test_model.py
def test_predict_frame(model, sample_frame):
    """Test frame prediction returns valid result."""
    result = model.predict_frame(sample_frame)
    
    assert isinstance(result, dict)
    assert 'species' in result
    assert 'confidence' in result
    assert 0 <= result['confidence'] <= 1
```

### Frontend Tests
```javascript
// tests/CameraCapture.test.jsx
import { render, screen } from '@testing-library/react';
import CameraCapture from '../components/CameraCapture';

test('renders camera component', () => {
  render(<CameraCapture />);
  expect(screen.getByText(/Camera Feed/i)).toBeInTheDocument();
});
```

## Documentation

### Code Documentation
- Add docstrings to all Python functions/classes
- Add JSDoc comments to JavaScript functions
- Update README.md for major changes
- Add inline comments for complex logic

### API Documentation
- Update OpenAPI schema for new endpoints
- Add examples to endpoint descriptions
- Document request/response formats

## Performance Guidelines

### Backend
- Use async/await for I/O operations
- Implement caching where appropriate
- Profile code for bottlenecks
- Optimize database queries

### Frontend
- Minimize re-renders with React.memo
- Use lazy loading for components
- Optimize images and assets
- Implement virtual scrolling for long lists

## Security Guidelines

- Never commit API keys or secrets
- Validate all user inputs
- Use parameterized queries
- Implement rate limiting
- Follow OWASP best practices

## Questions?

- Open an issue for discussion
- Join our community chat
- Check existing documentation
- Review closed issues/PRs

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
