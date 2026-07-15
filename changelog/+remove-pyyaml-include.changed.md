- Removed the `pyyaml-include` core dependency (GPL-3.0 licensed) in favor of a
  small built-in `!include` constructor for eval scenario files. Scenario
  `!include` behavior is unchanged: paths still resolve relative to the scenario
  file's directory and nested includes still work. Pipecat's core install no
  longer pulls in any GPL-licensed package.
