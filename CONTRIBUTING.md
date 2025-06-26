## Contributing to Pipecat

We welcome contributions of all kinds! Your help is appreciated. Follow these steps to get involved:

1. **Fork this repository**: Start by forking the Pipecat Documentation repository to your GitHub account.

2. **Clone the repository**: Clone your forked repository to your local machine.
   ```bash
   git clone https://github.com/your-username/pipecat
   ```
3. **Create a branch**: For your contribution, create a new branch.
   ```bash
   git checkout -b your-branch-name
   ```
4. **Make your changes**: Edit or add files as necessary.
5. **Test your changes**: Ensure that your changes look correct and follow the style set in the codebase.
6. **Commit your changes**: Once you're satisfied with your changes, commit them with a meaningful message.

```bash
git commit -m "Description of your changes"
```

7. **Push your changes**: Push your branch to your forked repository.

```bash
git push origin your-branch-name
```

8. **Submit a Pull Request (PR)**: Open a PR from your forked repository to the main branch of this repo.
   > Important: Describe the changes you've made clearly!

Our maintainers will review your PR, and once everything is good, your contributions will be merged!

## Code Style and Documentation

### Python Code Style

We use Ruff for code linting and formatting. Please ensure your code passes all linting checks before submitting a PR.

### Docstring Conventions

We follow Google-style docstrings with these specific conventions:

**Regular Classes:**

- Class docstring describes the class purpose and documents all `__init__` parameters in an `Args:` section
- No separate `__init__` docstring needed
- All public methods must have docstrings with `Args:` and `Returns:` sections as appropriate

**Dataclasses:**

- Class docstring describes the purpose and documents all fields in a `Parameters:` section
- No `__init__` docstring (auto-generated)

**Properties:**

- Must have docstrings with `Returns:` section

**Abstract Methods:**

- Must have docstrings explaining what subclasses should implement

#### Examples:

```python
# Regular class
class MyService(BaseService):
    """Description of what the service does.

    Args:
        param1: Description of param1.
        param2: Description of param2. Defaults to True.
        **kwargs: Additional arguments passed to parent.
    """

    def __init__(self, param1: str, param2: bool = True, **kwargs):
        # No docstring - parameters documented above
        super().__init__(**kwargs)

    @property
    def sample_rate(self) -> int:
        """Get the current sample rate.

        Returns:
            The sample rate in Hz.
        """
        return self._sample_rate

    async def process_data(self, data: str) -> bool:
        """Process the provided data.

        Args:
            data: The data to process.

        Returns:
            True if processing succeeded.
        """
        pass

# Dataclass
@dataclass
class ConfigParams:
    """Configuration parameters for the service.

    Parameters:
        host: The host address.
        port: The port number. Defaults to 8080.
        timeout: Connection timeout in seconds.
    """

    host: str
    port: int = 8080
    timeout: float = 30.0
```

# Contributor Covenant Code of Conduct

## Our Pledge

We as members, contributors, and leaders pledge to make participation in our
community a harassment-free experience for everyone, regardless of age, body
size, visible or invisible disability, ethnicity, sex characteristics, gender
identity and expression, level of experience, education, socio-economic status,
nationality, personal appearance, race, caste, color, religion, or sexual
identity and orientation.

We pledge to act and interact in ways that contribute to an open, welcoming,
diverse, inclusive, and healthy community.

## Our Standards

Examples of behavior that contributes to a positive environment for our
community include:

- Demonstrating empathy and kindness toward other people
- Being respectful of differing opinions, viewpoints, and experiences
- Giving and gracefully accepting constructive feedback
- Accepting responsibility and apologizing to those affected by our mistakes,
  and learning from the experience
- Focusing on what is best not just for us as individuals, but for the overall
  community

Examples of unacceptable behavior include:

- The use of sexualized language or imagery, and sexual attention or advances of
  any kind
- Trolling, insulting or derogatory comments, and personal or political attacks
- Public or private harassment
- Publishing others' private information, such as a physical or email address,
  without their explicit permission
- Other conduct which could reasonably be considered inappropriate in a
  professional setting

## Enforcement Responsibilities

Community leaders are responsible for clarifying and enforcing our standards of
acceptable behavior and will take appropriate and fair corrective action in
response to any behavior that they deem inappropriate, threatening, offensive,
or harmful.

Community leaders have the right and responsibility to remove, edit, or reject
comments, commits, code, wiki edits, issues, and other contributions that are
not aligned to this Code of Conduct, and will communicate reasons for moderation
decisions when appropriate.

## Scope

This Code of Conduct applies within all community spaces, and also applies when
an individual is officially representing the community in public spaces.
Examples of representing our community include using an official email address,
posting via an official social media account, or acting as an appointed
representative at an online or offline event.

## Enforcement

Instances of abusive, harassing, or otherwise unacceptable behavior may be
reported to the community leaders responsible for enforcement at pipecat-ai@daily.co.
All complaints will be reviewed and investigated promptly and fairly.

All community leaders are obligated to respect the privacy and security of the
reporter of any incident.

## Enforcement Guidelines

Community leaders will follow these Community Impact Guidelines in determining
the consequences for any action they deem in violation of this Code of Conduct:

### 1. Correction

**Community Impact**: Use of inappropriate language or other behavior deemed
unprofessional or unwelcome in the community.

**Consequence**: A private, written warning from community leaders, providing
clarity around the nature of the violation and an explanation of why the
behavior was inappropriate. A public apology may be requested.

### 2. Warning

**Community Impact**: A violation through a single incident or series of
actions.

**Consequence**: A warning with consequences for continued behavior. No
interaction with the people involved, including unsolicited interaction with
those enforcing the Code of Conduct, for a specified period of time. This
includes avoiding interactions in community spaces as well as external channels
like social media. Violating these terms may lead to a temporary or permanent
ban.

### 3. Temporary Ban

**Community Impact**: A serious violation of community standards, including
sustained inappropriate behavior.

**Consequence**: A temporary ban from any sort of interaction or public
communication with the community for a specified period of time. No public or
private interaction with the people involved, including unsolicited interaction
with those enforcing the Code of Conduct, is allowed during this period.
Violating these terms may lead to a permanent ban.

### 4. Permanent Ban

**Community Impact**: Demonstrating a pattern of violation of community
standards, including sustained inappropriate behavior, harassment of an
individual, or aggression toward or disparagement of classes of individuals.

**Consequence**: A permanent ban from any sort of public interaction within the
community.

## Attribution

This Code of Conduct is adapted from the [Contributor Covenant][homepage],
version 2.1, available at
[https://www.contributor-covenant.org/version/2/1/code_of_conduct.html][v2.1].

Community Impact Guidelines were inspired by
[Mozilla's code of conduct enforcement ladder][Mozilla CoC].

For answers to common questions about this code of conduct, see the FAQ at
[https://www.contributor-covenant.org/faq][FAQ]. Translations are available at
[https://www.contributor-covenant.org/translations][translations].

[homepage]: https://www.contributor-covenant.org
[v2.1]: https://www.contributor-covenant.org/version/2/1/code_of_conduct.html
[Mozilla CoC]: https://github.com/mozilla/diversity
[FAQ]: https://www.contributor-covenant.org/faq
[translations]: https://www.contributor-covenant.org/translations
