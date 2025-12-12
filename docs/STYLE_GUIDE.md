# Documentation Style Guide

This style guide ensures consistency across all documentation in the Physical AI & Humanoid Robotics Book.

## Writing Style

### Audience
- Write for beginner-to-intermediate developers
- Avoid jargon when simpler terms suffice
- Explain technical concepts with analogies when helpful
- Assume readers have basic programming knowledge but may be new to robotics

### Tone
- Use clear, concise language
- Be encouraging and supportive
- Avoid condescending language
- Maintain a professional but approachable tone

### Structure
- Start each section with learning objectives
- End with a summary and practice tasks
- Use consistent headings and formatting
- Include examples to illustrate concepts

## Markdown Formatting

### Headings
```markdown
# Module Title (H1 - only one per document)
## Section Title (H2)
### Subsection Title (H3)
#### Minor Section (H4 if needed)
```

### Code Blocks
- Always specify the language for syntax highlighting
- Use descriptive variable names in examples
- Include comments for complex code
- Show expected output when relevant

```markdown
```python
# Example Python code
def example_function():
    """Brief description of what the function does."""
    pass
```
```

### Lists
- Use unordered lists for items without sequence
- Use ordered lists for step-by-step procedures
- Use task lists for learning objectives or checklists

### Emphasis
- Use **bold** for important terms or actions
- Use *italics* for new terms or emphasis
- Use `code` formatting for file names, commands, and variables

## Technical Documentation Standards

### Code Examples
- Include complete, copy-paste ready code samples
- Add comments to explain complex concepts
- Test all examples in clean environments
- Follow consistent formatting and naming conventions
- Include error handling where appropriate

### Images and Diagrams
- Include descriptive alt text for accessibility
- Use consistent naming conventions
- Store in appropriate module asset directories
- Provide attribution when required

### Links
- Use descriptive link text instead of URLs
- Link to official documentation when available
- Verify all links work before publication

## Content Organization

### Module Structure
Each module should follow this pattern:
1. Introduction with learning objectives
2. Prerequisites
3. Content sections with examples
4. Mini-project or practical application
5. Troubleshooting section
6. Summary and next steps

### Section Structure
Each content section should include:
- Clear heading
- Brief introduction
- Detailed explanation with examples
- Key takeaways or summary points

## Quality Standards

### Accuracy
- Verify all technical information against official documentation
- Test all code examples in actual environments
- Update content when APIs or tools change
- Include version information when relevant

### Reproducibility
- All instructions must work on Ubuntu 22.04, Windows, and macOS
- Include full command examples
- Specify required software versions
- Provide troubleshooting guidance

### Clarity
- Use active voice when possible
- Break complex concepts into digestible parts
- Provide context before diving into details
- Use consistent terminology throughout

## Review Checklist

Before finalizing any documentation, ensure:
- [ ] Learning objectives are clearly stated
- [ ] Content is appropriate for target audience
- [ ] All code examples are tested and functional
- [ ] Technical information is accurate
- [ ] Links are valid and properly formatted
- [ ] Images have appropriate alt text
- [ ] Style guide standards are followed
- [ ] Grammar and spelling are correct