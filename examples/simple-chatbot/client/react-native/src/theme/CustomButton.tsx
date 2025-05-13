import React from 'react';
import { TouchableOpacity, Text, StyleSheet, ViewStyle, TextStyle, GestureResponderEvent } from 'react-native';
import { MaterialIcons } from '@expo/vector-icons';

interface CustomButtonProps {
  title: string;
  onPress: (event: GestureResponderEvent) => void;
  backgroundColor?: string; // Optional prop for background color
  textColor?: string; // Optional prop for text color
  style?: ViewStyle; // Optional additional styles for the button container
  textStyle?: TextStyle; // Optional additional styles for the text
  iconName?: string; // Optional prop for the icon name from MaterialIcons
  iconPosition?: 'left' | 'right'; // Optional prop to control icon position
  iconSize?: number; // Optional prop for icon size
  iconColor?: string; // Optional prop for icon color
}

const CustomButton: React.FC<CustomButtonProps> = ({
  title,
  onPress,
  backgroundColor = 'black',
  textColor = 'white',
  style,
  textStyle,
  iconName,
  iconPosition = 'left',
  iconSize = 24,
  iconColor = 'white',
}) => {
  return (
    <TouchableOpacity
      onPress={onPress}
      style={[styles.button, { backgroundColor }, style]}>
      {iconName && iconPosition === 'left' && (
        <MaterialIcons name={iconName as keyof typeof MaterialIcons.glyphMap} size={iconSize} color={iconColor} style={styles.icon} />
      )}
      <Text style={[styles.text, { color: textColor }, textStyle]}>{title}</Text>
      {iconName && iconPosition === 'right' && (
        <MaterialIcons name={iconName as keyof typeof MaterialIcons.glyphMap} size={iconSize} color={iconColor} style={styles.icon} />
      )}
    </TouchableOpacity>
  );
};

const styles = StyleSheet.create({
  button: {
    padding: 12,
    borderRadius: 8,
    alignItems: 'center',
    justifyContent: 'center',
    flexDirection: 'row', // Ensures icon and text are aligned in a row
  },
  text: {
    fontSize: 16,
    fontWeight: 'bold',
  },
  icon: {
    marginHorizontal: 5, // Adds space between the icon and text
  },
});

export default CustomButton;
